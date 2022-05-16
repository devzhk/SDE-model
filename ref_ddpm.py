import os
import random
from tqdm import tqdm
from argparse import ArgumentParser
import yaml

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from models.ddpm import Generator, DDPM
from utils.dataset import myOdeData


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


def loss_fn(model, x, eps=1e-3):
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


def train(model, dataloader,
          optimizer, scheduler,
          device):
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
            batch = batch[:, -1].to(device)
            model.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)
        pbar.set_description(
            (
                f'Epoch: {e}; Loss: {train_loss}'
            )
        )
        if e % save_step:
            torch.save(model.state_dict(), f'{save_ckpt_dir}/ddpm-{e}.pt')
    torch.save(model.state_dict(), f'{save_ckpt_dir}/ddpm-final.pt')


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
    train(model, train_loader,
          optimizer, scheduler,
          device)
