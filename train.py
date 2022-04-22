import os
import math
import yaml
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import MultiStepLR
from utils.optim import Adam


from tqdm import tqdm

from models.fno import FNN2d, FNN1d

from utils.helper import kde, group_kde
from utils.dataset import myOdeData, get_init



def train(model, criterion,
          optimizer, scheduler,
          dataloader, device, config):
    t_dim = config['t_dim']
    t_step = config['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['logname']
    save_step = config['save_step']
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
            states = states.to(device)

            pred = model(in_state)
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

        if e % save_step == 0:
            if dimension == 1:
                zip_state = [pred[:, -1, 0].detach().numpy(), states[:, -1, 0].detach().numpy()]
                labels = ['Prediction', 'Truth']
                group_kde(zip_state, labels, f'{save_img_dir}/train_{e}.png')
            elif dimension == 2:
                kde(pred[:, -1, :], save_file=f'{save_img_dir}/train_{e}_pred.png', dim=2)
                kde(states[:, -1, :], save_file=f'{save_img_dir}/train_{e}_truth.png', dim=2)
            # kde(pred[:, -1, 0].detach().numpy(), f'figs/1dGM/pred_{e}.png')
            # kde(states[:, -1, 0].detach().numpy(), f'figs/1dGM/true_{e}.png')
            torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')

    if dimension == 1:
        zip_state = [pred[:, -1, 0].detach().numpy(), states[:, -1, 0].detach().numpy()]
        labels = ['Prediction', 'Truth']
        group_kde(zip_state, labels, f'{save_img_dir}/train_final.png')
    elif dimension == 2:
        kde(pred[:, -1, :], save_file=f'{save_img_dir}/train_final_pred.png', dim=2)
        kde(states[:, -1, :], save_file=f'{save_img_dir}/train_final_truth.png', dim=2)

    torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/train_2d.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # parse configuration file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epoch = config['num_epoch']
    batchsize = config['batchsize']
    dimension = config['dimension']
    
    #
    dataset = myOdeData(config['datapath'], config['t_step'])
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    model = FNN1d(modes=config['modes'],
                  fc_dim=config['fc_dim'],
                  layers=config['layers'],
                  in_dim=dimension + 1, out_dim=dimension,
                  activation=config['activation']).to(device)
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)
    criterion = nn.MSELoss()
    train(model, criterion,
          optimizer, scheduler,
          train_loader, device, config)

