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

from models.fcn import FCN

from utils.helper import kde, group_kde


class myOdeData(Dataset):
    def __init__(self, datapath, t_step, num_sample=None):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        data = raw['data'].detach().clone()
        if num_sample is None:
            self.data = data[:, 0::t_step]
        else:
            self.data = data[:num_sample, 0::t_step]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def train(model, dataloader,
          criterion,
          optimizer, scheduler,
          device, config):
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


    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)
    for e in pbar:
        train_loss = 0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:, 0]
            out_state = states[:, -1]
            pred = model(in_state)
            loss = criterion(pred, out_state)
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
                zip_state = [pred.detach().numpy(), out_state.detach().numpy()]
                labels = ['Prediction', 'Truth']
                group_kde(zip_state, labels, f'{save_img_dir}/train_{e}.png')
            elif dimension == 2:
                kde(pred, save_file=f'{save_img_dir}/train_{e}_pred.png', dim=2)
                kde(out_state, save_file=f'{save_img_dir}/train_{e}_truth.png', dim=2)
            # kde(pred[:, -1, 0].detach().numpy(), f'figs/1dGM/pred_{e}.png')
            # kde(states[:, -1, 0].detach().numpy(), f'figs/1dGM/true_{e}.png')
            torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')

    if dimension == 1:
        zip_state = [pred[:, -1, 0].detach().numpy(), out_state[:, -1, 0].detach().numpy()]
        labels = ['Prediction', 'Truth']
        group_kde(zip_state, labels, f'{save_img_dir}/train_final.png')
    elif dimension == 2:
        kde(pred, save_file=f'{save_img_dir}/train_final_pred.png', dim=2)
        kde(out_state, save_file=f'{save_img_dir}/train_final_truth.png', dim=2)

    torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/train_2d-fcn.yaml', help='configuration file')
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
    num_sample = config['num_sample'] if 'num_sample' in config else None
    dataset = myOdeData(config['datapath'], config['t_step'], num_sample)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    model = FCN(layers=config['layers'], activation=config['activation']).to(device)
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)
    criterion = nn.MSELoss()
    train(model, train_loader,
          criterion,
          optimizer, scheduler,
          device, config)

