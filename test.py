import os
import numpy as np
import yaml
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN1d

from utils.helper import kde, group_kde


def get_init(x, ts):
    '''
    x: (batchsize, t_dim, 1)
    ts: (t_dim, )
    '''
    t_data = ts.repeat(x.shape[0], 1).unsqueeze(-1)
    return torch.cat([x, t_data], dim=-1)


class myOdeData(Dataset):
    def __init__(self, datapath):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        self.data = raw['data'].detach().clone()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def eval(model, dataloader, criterion, 
         device, config):
    logname = config['logname']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    t0, t1 = 1., config['epsilon']
    ts = torch.linspace(t0, t1, t_dim)
    model.eval()
    pred_list = []
    truth_list = []
    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            ini_state = states[:, 0:1, :].repeat(1, t_dim, 1)
            in_state = get_init(ini_state, ts).to(device)
            states = states.to(device)
            pred = model(in_state)

            pred_list.append(pred)
            truth_list.append(states)
            loss = criterion(pred, states)

            test_err += loss.item()
    
    test_err /= len(dataloader)
    print(f'Test MSE of the whole trajectory: {test_err}')
    final_pred = torch.cat(pred_list, dim=0)
    final_states = torch.cat(truth_list, dim=0)

    err_T = criterion(final_pred[:, -1, :], final_states[:, -1, :])
    print(f'Test MSE at time 0: {err_T}')

    if dimension == 1:
        zip_state = [final_pred[:, -1, 0].detach().numpy(), final_states[:, -1, 0].detach().numpy()]
        labels = ['Prediction', 'Truth']
        group_kde(zip_state, labels, f'{save_img_dir}/test.png')
    elif dimension == 2:
        kde(final_pred[:, -1, :], save_file=f'{save_img_dir}/test_pred.png', dim=2)
        kde(final_states[:, -1, :], save_file=f'{save_img_dir}/test_truth.png', dim=2)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/test_2d.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # parse configuration file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchsize = config['batchsize']
    t_dim = config['t_dim']
    dimension = config['dimension']
    ckpt_path = config['ckpt']

    # 
    dataset = myOdeData(config['datapath'])
    test_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    model = FNN1d(modes=config['modes'],
                  fc_dim=config['fc_dim'],
                  layers=config['layers'],
                  in_dim=dimension + 1, out_dim=dimension,
                  activation=config['activation']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    criterion = nn.MSELoss()
    eval(model, test_loader, criterion, device, config)