import math
import os
import random
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.unet3d import Unet3D
from utils.data_helper import data_sampler
from utils.dataset import ImageData
from utils.distributed import setup, cleanup, reduce_loss_dict, all_reduce_mean
from utils.helper import count_params

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

    # t0, t1 = 1., config['epsilon']
    model.eval()
    pred_list = []
    truth_list = []

    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:, :, 0:1].repeat(1, 1, num_t, 1, 1)
            # in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)

            pred = model(in_state)
            if num_pad > 0:
                pred = pred[:, :-num_pad, :]
            pred_list.append(pred[:, :, -1])
            truth_list.append(states[:, :, -1])
            loss = criterion(pred, states)

            test_err += loss
    test_err /= len(dataloader)
    final_pred = torch.cat(pred_list, dim=0)
    final_states = torch.cat(truth_list, dim=0)
    err_T = criterion(final_pred, final_states)
    if device == 0:
        print(f'MSE of the whole trajectory: {test_err}')
        print(f'MSE at time 0: {err_T}')
        save_image(pred[:, :, -1],
                   f'{save_img_dir}/pred_test.png',
                   normalize=True,
                   value_range=(-1, 1))
        save_image(states[:, :, -1],
                   f'{save_img_dir}/truth_test.png',
                   normalize=True,
                   value_range=(-1, 1))
    return err_T


def train(model, dataloader,
          criterion,
          optimizer, scheduler,
          device, config, args,
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

    # t0, t1 = 1., config['epsilon']
    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)

    for e in pbar:
        log_dict = {
            'train_loss': 0.0,
            'test_error': 0.0
        }
        if e < 20:
            ratio = 1.
        elif e < 200:
            ratio = 10 / e
        else:
            ratio = 0.0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:, :, 0:1].repeat(1, 1, num_t, 1, 1)
            # in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)

            pred = model(in_state, prob_focus_present=ratio)
            if num_pad > 0:
                pred = pred[:, :-num_pad, :]
            loss = criterion(pred, states)
            # update model
            model.zero_grad()
            loss.backward()
            optimizer.step()
            log_dict['train_loss'] += loss
        scheduler.step()

        reduced_log_dict = reduce_loss_dict(log_dict)
        train_loss = reduced_log_dict['train_loss'].item() / len(dataloader)
        if device == 0:
            pbar.set_description(
                (
                    f'Epoch :{e}, Loss: {train_loss}'
                )
            )
        log_state = {
            'train MSE': train_loss
        }

        if e % save_step == 0:
            if device == 0:
                save_image(pred[:, :, -1],
                           f'{save_img_dir}/pred_{e}.png',
                           normalize=True,
                           value_range=(-1, 1))
                save_image(states[:, :, -1],
                           f'{save_img_dir}/truth_{e}.png',
                           normalize=True,
                           value_range=(-1, 1))
                torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')
            # eval on validation set
            if valloader:
                val_err = eval(model, valloader, criterion,
                               device, config, True)
                val_err_avg = all_reduce_mean(val_err)
                log_state['val MSE'] = val_err_avg
        if use_wandb and wandb:
            wandb.log(log_state)
    torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')
    # test
    test_err = eval(model, testloader, criterion, device, config)
    test_err_avg = all_reduce_mean(test_err).item()

    if use_wandb and wandb:
        wandb.log({
            'test MSE': test_err_avg
        })
        run.finish()
    if device == 0:
        print(f'test MSE : {test_err_avg}')
    if args.distributed:
        cleanup()
    print(f'Process {device} exits...')


def run(train_loader, val_loader, test_loader,
        config, args, device):
    # create model
    model = Unet3D(dim=48, no_skipped=1).to(device)
    num_params = count_params(model)
    print(f'number of parameters: {num_params}')
    config['num_params'] = num_params

    if args.distributed:
        model = DDP(model, device_ids=[device], broadcast_buffers=False)
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['milestone'],
                            gamma=0.5)
    criterion = nn.MSELoss()
    train(model, train_loader,
          criterion,
          optimizer, scheduler,
          device, config, args,
          valloader=test_loader,
          testloader=test_loader)


def subprocess_fn(rank, args):
    # setup
    if args.distributed:
        setup(rank, args.num_gpus, port=f'{args.port}')
    print(f'Running on rank: {rank}')

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    # parse configuration file
    device = rank
    batchsize = config['batchsize']
    if args.log and rank == 0:
        config['use_wandb'] = True
    else:
        config['use_wandb'] = False

    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # training set
    num_sample = config['num_sample'] if 'num_sample' in config else None
    trainset = ImageData(config['datapath'], config['t_step'], num_sample)
    train_loader = DataLoader(trainset, batch_size=batchsize,
                              sampler=data_sampler(trainset,
                                                   shuffle=True,
                                                   distributed=args.distributed),
                              drop_last=True)
    # validation set
    # num_val_data = config['num_val_data'] if 'num_val_data' in config else None
    # valset = ImageData(config['val_datapath'], config['t_step'], num_val_data)
    # val_loader = DataLoader(valset, batch_size=batchsize, shuffle=True)
    # test set
    testset = ImageData(config['test_datapath'], config['t_step'])
    test_loader = DataLoader(testset, batch_size=batchsize,
                             sampler=data_sampler(testset,
                                                  shuffle=True,
                                                  distributed=args.distributed),
                             drop_last=True)

    for i in range(args.repeat):
        run(train_loader, test_loader, test_loader, config, args, device)
    print(f'{args.repeat} runs done!')
    if args.distributed:
        cleanup()


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/cifar/train_unet3d_s9.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--port', type=str, default='9037')
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    if args.seed is None:
        args.seed = random.randint(0, 100000)
    if args.distributed:
        mp.spawn(subprocess_fn, args=(args,), nprocs=args.num_gpus)
    else:
        subprocess_fn(0, args)
