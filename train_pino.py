import os
import math
import random
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.optim import Adam

from models.fno import FNN1d

from utils.helper import kde, count_params
from utils.dataset import myOdeData, get_init
from utils.loss import ode_loss, fdm_ode_loss

try:
    import wandb
except ImportError:
    wandb = None


class Gmm_score(object):
    def __init__(self, mu=2.0, num_modes=2, sigma2=0.01, beta_min=0.1, beta_max=20., dim=2):
        if dim == 1:
            self.mus = torch.linspace(-mu, mu, num_modes).unsqueeze(-1)
        elif dim == 2:
            xs = torch.linspace(-mu, mu, int(np.sqrt(num_modes)))
            ys = torch.linspace(-mu, mu, int(np.sqrt(num_modes)))
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            self.mus = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
        print(f'mu: {mu}, num_modes: {num_modes}, sigma2: {sigma2}')

        assert len(self.mus) == num_modes
        self.num_modes = num_modes

        self.sigma2 = sigma2
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alphas_t = lambda t: torch.exp(-0.5 *
                                            (beta_max - beta_min) * t**2 - beta_min * t)

    def score(self, xs, ts):
        # ts: (B, 1)
        # xs: (B, d)

        d = self.mus.shape[1]
        M = self.mus.shape[0]
        B = xs.shape[0]
        mus = self.mus.to(xs.device).reshape(1, M, d)
        alpha_t = self.alphas_t(ts).reshape(B, 1, 1)
        mus_t = mus * torch.sqrt(alpha_t)   # (B, M, d)
        sigma2_t = 1 - (1 - self.sigma2) * alpha_t  # (B, 1, 1)

        xs = xs.reshape(B, 1, d)
        diff = xs - mus_t   # (B, M, d)
        logits_t = - torch.sum(diff * diff, dim=-1) / \
                   (2 * sigma2_t.squeeze(-1))  # (B, M)
        gmm_coef_t = F.softmax(logits_t, dim=1)  # (B, M)

        sum_mu = torch.einsum('bm,bmd->bd', gmm_coef_t, mus_t)  # (B, d)
        xs = xs.reshape(B, d)
        score = (sum_mu - xs) / sigma2_t.reshape(B, 1)
        return score


class VPODE(object):
    def __init__(self, model, beta_min, beta_max, scale_sigma=False):
        """Construct a Variance Preserving SDE.
        Args:
          model: diffusion model
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        self.model = model
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alphas_t = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t ** 2 - beta_min * t)
        self.scale_sigma = scale_sigma

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def ode_fn(self, t, x):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)
        # ts = t.repeat(x.shape[0])
        score = self.model.score(x, t)
        if self.scale_sigma:
            score = - score / torch.sqrt(1 - self.alphas_t(t))

        ode_coef = drift - 0.5 * diffusion ** 2 * score
        return ode_coef


def eval(model, dataloader, criterion,
         device, config, plot=False):
    t_dim = config['t_dim']
    t_step = config['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['logname']
    num_pad = config['num_pad'] if 'num_pad' in config else 5
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    t0, t1 = 1., config['epsilon']
    ts = torch.linspace(t0, t1, num_t)
    model.eval()
    pred_list = []
    truth_list = []
    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            ini_state = states[:, 0:1, :].repeat(1, num_t, 1)
            in_state = get_init(ini_state, ts).to(device)
            in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)
            states = states.to(device)
            pred = model(in_state)
            pred = pred[:, :-num_pad, :]
            pred_list.append(pred)
            truth_list.append(states)
            loss = criterion(pred, states)

            test_err += loss.item()

    test_err /= len(dataloader)
    print(f'MSE of the whole trajectory: {test_err}')
    final_pred = torch.cat(pred_list, dim=0)
    final_states = torch.cat(truth_list, dim=0)

    err_T = criterion(final_pred[:, -1, :], final_states[:, -1, :])
    print(f'MSE at time 0: {err_T}')
    if plot:
        kde(final_pred[:, -1, :], save_file=f'{save_img_dir}/test_pred.png', dim=2)
        kde(final_states[:, -1, :], save_file=f'{save_img_dir}/test_truth.png', dim=2)
    return err_T


def train(model, dataloader,
          criterion,
          optimizer, scheduler,
          device, config,
          ode,                    # ODE
          valloader=None,
          testloader=None):
    t_dim = config['t_dim']
    t_step = config['t_step']
    eq_weight = config['eq_weight']
    num_t = math.ceil(t_dim / t_step)
    num_pad = config['num_pad'] if 'num_pad' in config else 5
    logname = config['logname']
    save_step = config['save_step']
    use_wandb = config['use_wandb'] if 'use_wandb' in config else False
    if use_wandb and wandb:
        wandb.init(entity=config['entity'],
                   project=config['project'],
                   group=config['group'],
                   config=config)

    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    t0, t1 = 1., config['epsilon']
    ts = torch.linspace(t0, t1, num_t)
    L = (t0 - t1) * (num_pad + num_t - 1) / (num_t - 1)     # period of extended domain
    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)
    for e in pbar:
        train_loss = 0
        train_data = 0
        train_eq = 0
        for states in dataloader:
            ini_state = states[:, 0:1, :].repeat(1, num_t, 1)
            in_state = get_init(ini_state, ts).to(device)
            in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)
            states = states.to(device)

            out = model(in_state)
            pred = out[:, :-num_pad, :]
            data_loss = criterion(pred, states)
            eq_loss = ode_loss(out, ts.to(device),
                               ode, criterion,
                               num_pad=num_pad, L=L)
            # eq_loss = fdm_ode_loss(out[:, 1:-1, :], ts.to(device), ode, criterion)
            loss = data_loss + eq_loss * eq_weight

            # update model
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_data += data_loss.item()
            train_eq += eq_loss.item()
        scheduler.step()
        train_loss /= len(train_loader)
        train_data /= len(train_loader)
        train_eq /= len(train_loader)
        pbar.set_description(
            (
                f'Epoch :{e}, Loss: {train_loss}'
            )
        )
        log_state = {
            'train MSE': train_loss,
            'Train Data MSE': train_data,
            'Train ODE MSE': train_eq
        }

        if e % save_step == 0:
            kde(pred[:, -1, :], save_file=f'{save_img_dir}/train_{e}_pred.png', dim=2)
            kde(states[:, -1, :], save_file=f'{save_img_dir}/train_{e}_truth.png', dim=2)
            torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')
            # eval on validation set
            if valloader:
                print('start evaluating on validation set...')
                val_err = eval(model, valloader, criterion,
                               device, config)
                log_state['val MSE'] = val_err
        if use_wandb and wandb:
            wandb.log(log_state)
    torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')
    # test
    test_err = eval(model, testloader, criterion, device, config, plot=True)
    if use_wandb and wandb:
        wandb.log({
            'test MSE': test_err
        })
    print(f'test MSE : {test_err}')


def run(train_loader, val_loader, test_loader,
        config, device):
    # set random seed
    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config['seed'] = seed
    # create model
    model = FNN1d(modes=config['modes'],
                  fc_dim=config['fc_dim'],
                  layers=config['layers'],
                  in_dim=dimension + 1, out_dim=dimension,
                  activation=config['activation']).to(device)
    num_params = count_params(model)
    print(f'number of parameters: {num_params}')
    config['num_params'] = num_params
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)
    criterion = nn.MSELoss()
    gmm = Gmm_score(mu=2.0, sigma2=1e-4, num_modes=25)
    vpode = VPODE(gmm, beta_min=gmm.beta_min, beta_max=gmm.beta_max)
    train(model, train_loader,
          criterion,
          optimizer, scheduler,
          device, config,
          ode=vpode,
          valloader=test_loader,
          testloader=test_loader)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/train_2d-s9.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # parse configuration file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epoch = config['num_epoch']
    batchsize = config['batchsize']
    dimension = config['dimension']
    if args.log:
        config['use_wandb'] = True
    else:
        config['use_wandb'] = False
    # training set
    num_sample = config['num_sample'] if 'num_sample' in config else None
    trainset = myOdeData(config['datapath'], config['t_step'], num_sample)
    train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # validation set
    num_val_data = config['num_val_data'] if 'num_val_data' in config else None
    valset = myOdeData(config['val_datapath'], config['t_step'], num_val_data)
    val_loader = DataLoader(valset, batch_size=batchsize, shuffle=True)
    # test set
    testset = myOdeData(config['test_datapath'], config['t_step'])
    test_loader = DataLoader(testset, batch_size=batchsize, shuffle=True)

    for i in range(args.repeat):
        run(train_loader, test_loader, test_loader, config, device)
    print(f'{args.repeat} runs done!')
