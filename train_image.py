import math
import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet3d import Unet3D
from utils.dataset import myOdeData, get_init
from utils.helper import kde, count_params
from utils.optim import Adam

try:
    import wandb
except ImportError:
    wandb = None


def eval(model, dataloader, criterion,
         device, config, plot=False):
    t_dim = config['t_dim']
    t_step = config['t_step']
    num_t = math.ceil(t_dim / t_step)
    num_pad = config['num_pad'] if 'num_pad' in config else 5
    logname = config['logname']
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
            if num_pad > 0:
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
          valloader=None,
          testloader=None):
    t_dim = config['t_dim']
    t_step = config['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['logname']
    num_pad = config['num_pad'] if 'num_pad' in config else 5
    save_step = config['save_step']
    use_wandb = config['use_wandb'] if 'use_wandb' in config else False
    if use_wandb and wandb:
        run = wandb.init(entity=config['entity'],
                         project=config['project'],
                         group=config['group'],
                         config=config,
                         reinit=True,
                         settings=wandb.Settings(start_method='fork'))

    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    t0, t1 = 1., config['epsilon']
    ts = torch.linspace(t0, t1, num_t)
    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)
    for e in pbar:
        train_loss = 0
        for states in dataloader:
            ini_state = states[:, 0:1, :].repeat(1, num_t, 1)
            in_state = get_init(ini_state, ts).to(device)
            in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)
            states = states.to(device)

            pred = model(in_state)
            if num_pad > 0:
                pred = pred[:, :-num_pad, :]
            loss = criterion(pred, states)
            # update model
            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)
        pbar.set_description(
            (
                f'Epoch :{e}, Loss: {train_loss}'
            )
        )
        log_state = {
            'train MSE': train_loss
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
        run.finish()

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
    model = Unet3D(dim=32).to(device)
    num_params = count_params(model)
    print(f'number of parameters: {num_params}')
    config['num_params'] = num_params
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)
    criterion = nn.MSELoss()
    train(model, train_loader,
          criterion,
          optimizer, scheduler,
          device, config,
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
