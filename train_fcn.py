import os
import math
import random
import yaml
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import MultiStepLR
from utils.optim import Adam


from tqdm import tqdm

from models.fcn import FCN

from utils.helper import kde, group_kde, count_params
from utils.dataset import myOdeData

try:
    import wandb
except ImportError:
    wandb = None


def eval(model, dataloader, criterion,
         device, config, plot=False):
    logname = config['logname']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    model.eval()
    pred_list = []
    truth_list = []
    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:, 0]
            out_state = states[:, -1]
            pred = model(in_state)

            pred_list.append(pred)
            truth_list.append(out_state)
            loss = criterion(pred, out_state)

            test_err += loss.item()

    test_err /= len(dataloader)
    final_pred = torch.cat(pred_list, dim=0)
    final_states = torch.cat(truth_list, dim=0)

    err_T = criterion(final_pred, final_states)
    print(f'MSE at time 0: {err_T}')

    if plot:
        kde(final_pred, save_file=f'{save_img_dir}/test_pred.png', dim=2)
        kde(final_states, save_file=f'{save_img_dir}/test_truth.png', dim=2)
    return err_T


def train(model, dataloader,
          criterion,
          optimizer, scheduler,
          device, config,
          valloader=None,
          testloader=None):
    # wandb initialization
    use_wandb = config['use_wandb'] if 'use_wandb' in config else False
    if use_wandb and wandb:
        wandb.init(entity=config['entity'],
                   project=config['project'],
                   group=config['group'],
                   config=config)

    logname = config['logname']
    save_step = config['save_step']

    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)
    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # training
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
        log_state = {
            'train MSE': train_loss
        }

        if e % save_step == 0:
            kde(pred, save_file=f'{save_img_dir}/train_{e}_pred.png', dim=2)
            kde(out_state, save_file=f'{save_img_dir}/train_{e}_truth.png', dim=2)
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


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/gaussian/train_2d-fcn.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    # set random seed
    seed = random.randint(0, 100000)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config['seed'] = seed
    # parse configuration file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epoch = config['num_epoch']
    batchsize = config['batchsize']
    dimension = config['dimension']

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

    # create model
    model = FCN(layers=config['layers'], activation=config['activation']).to(device)
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

