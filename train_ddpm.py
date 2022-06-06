import os
from tqdm import tqdm
from argparse import ArgumentParser
import yaml

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import Dataset, DataLoader

from utils.sde_lib import VPSDE
from models.toy_ddpm import DDPM


def marginal_prob(x, t, beta_min=0.1, beta_max=20):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.

    Returns:
      The standard deviation.
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean = torch.exp(log_mean_coeff[:, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


def sde_coeff(x, t, beta_min=0.1, beta_max=20):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    beta_t = beta_min + t * (beta_max - beta_min)
    drift = -0.5 * beta_t[:, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion


def loss_fn2(model, x, eps=1e-4):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    mean, std = marginal_prob(x, random_t)

    perturbed_x = mean + z * std[:, None]
    score = model(perturbed_x, random_t)

    losses = torch.square(score - z)
    # losses = torch.square(score*std[:,None] + z)
    losses = torch.sum(losses, dim=1)  # * weight
    loss = torch.mean(losses)
    return loss


class myOdeData(Dataset):
    def __init__(self, datapath, num_sample=None):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        data = raw['data'].detach().clone()
        if num_sample is None:
            self.data = data[:, -1]
        else:
            self.data = data[:num_sample, -1]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def loss_fn(vpsde, model, batch):
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None] * noise
    score = model(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss


def train(model, dataloader,
          optimizer, scheduler,
          vpsde,
          config):
    logname = config['logname']
    save_step = config['save_step']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)
    for e in pbar:
        train_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            model.zero_grad()
            loss = loss_fn(vpsde, model, batch)
            # loss = loss_fn2(model, batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader)
        pbar.set_description(
            (
                f'Epoch: {e}; {train_loss}'
            )
        )
        scheduler.step()
        if e % save_step == 0:
            torch.save(model.state_dict(), f'{save_ckpt_dir}/score_{e}.pt')
    torch.save(model.state_dict(), f'{save_ckpt_dir}/score_final.pt')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/ddpm.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # parse configuration file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epoch = config['num_epoch']
    batchsize = config['batchsize']
    num_sample = config['num_sample'] if 'num_sample' in config else None
    dataset = myOdeData(config['datapath'], num_sample)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    model = DDPM(config).to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)

    sde = VPSDE(beta_min=config['beta_min'], beta_max=config['beta_max'], N=config['num_scales'])
    sampling_eps = 1e-4
    train(model, train_loader, optimizer, scheduler, sde, config)
