import math
import os
import random
from argparse import ArgumentParser
import copy
import psutil

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, LinearLR

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.tunet import TUnet
from models.utils import save_ckpt, interpolate_model
from utils.data_helper import data_sampler, sample_data
from utils.dataset import H5Data, split_list
from utils.distributed import setup, cleanup, reduce_loss_dict, all_reduce_mean
from utils.helper import count_params, dict2namespace

from eval_fid import compute_fid

try:
    import wandb
except ImportError:
    wandb = None


def eval(model, dataloader, criterion,
         device, config, epoch=-1):
    t_dim = config['data']['t_dim']
    t_step = config['data']['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['log']['logname']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    t0, t1 = 1., config['data']['epsilon']
    timesteps = torch.linspace(t0, t1, num_t, device=device)
    model.eval()

    with torch.no_grad():
        test_err = 0
        for states in dataloader:
            states = states.to(device)
            in_state = states[:,:,0]

            pred = model(in_state, timesteps)
            loss = criterion(pred, states)

            test_err += loss
    test_err /= len(dataloader)
    if device == 0:
        print(f'MSE of the whole trajectory: {test_err}')
        save_image(pred[:, :, -1] * 0.5 + 0.5,
                   f'{save_img_dir}/pred_test_{epoch}.png')
        save_image(states[:, :, -1] * 0.5 + 0.5,
                   f'{save_img_dir}/truth_test_{epoch}.png')
    return test_err


def train(model, model_ema,
          dataloader,
          criterion,
          optimizer, scheduler,
          device, config, args,
          valloader=None):
    # get configuration
    t_dim = config['data']['t_dim']
    t_step = config['data']['t_step']
    ema_decay = config['model']['ema_rate']
    start_iter = config['training']['start_iter']
    num_t = math.ceil(t_dim / t_step)
    logname = config['log']['logname']
    save_step = config['eval']['save_step']
    use_wandb = config['use_wandb'] if 'use_wandb' in config else False

    # setup wandb
    if use_wandb and wandb:
        run = wandb.init(entity=config['log']['entity'],
                         project=config['log']['project'],
                         group=config['log']['group'],
                         config=config,
                         reinit=True,
                         settings=wandb.Settings(start_method='fork'))

    # prepare exp dir
    base_dir = f'exp/{logname}'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # prepare time input
    t0, t1 = 1., config['data']['epsilon']
    timesteps = torch.linspace(t0, t1, num_t, device=device)

    # training
    model.train()
    pbar = tqdm(range(config['training']['n_iters']), dynamic_ncols=True)
    use_time_conv = config['model']['time_conv']
    log_dict = {}
    dataloader = sample_data(dataloader)

    for e in pbar:
        states = next(dataloader)
        states = states.to(device)
        in_state = states[:, :, 0]

        pred = model(in_state, timesteps)
        if use_time_conv:
            loss = criterion(pred[:, :, -1], states[:, :, -1])
        else:
            loss = criterion(pred, states)
        # update model
        model.zero_grad()
        loss.backward()
        optimizer.step()
        log_dict['train_loss'] = loss
        scheduler.step()

        reduced_log_dict = reduce_loss_dict(log_dict)
        train_loss = reduced_log_dict['train_loss'].item()
        if device == 0:
            pbar.set_description(
                (
                    f'Epoch :{e}, Loss: {train_loss}'
                )
            )
        log_state = {
            'train MSE': train_loss
        }

        # Update moving average of the model parameters
        interpolate_model(model_ema, model, beta=ema_decay)

        if e % save_step == 0:
            # memory_use = psutil.Process().memory_info().rss / (1024 * 1024)
            # print(f'Step {e}; Memory usage: {memory_use} MB')
            if device == 0:
                save_image(pred[:, :, -1] * 0.5 + 0.5,
                           f'{save_img_dir}/pred_{e}.png',
                           nrow=8)
                save_image(states[:, :, -1] * 0.5 + 0.5,
                           f'{save_img_dir}/truth_{e}.png',
                           nrow=8)
                save_path = os.path.join(save_ckpt_dir,
                                         f'solver-model_{start_iter + e}.pt')
                save_ckpt(save_path,
                          model=model, model_ema=model_ema,
                          optim=optimizer, args=args)
                if config['eval']['test_fid'] and e > 0:
                    fid_score = compute_fid(model_ema, t0, t1, num_t, device=device)
                    log_state['FID'] = fid_score
            # eval on validation set
            if valloader:
                val_err = eval(model_ema, valloader, criterion,
                               device, config, epoch=e)
                val_err_avg = all_reduce_mean(val_err)
                log_state['val MSE'] = val_err_avg
        if use_wandb and wandb:
            wandb.log(log_state)

    if device == 0:
        save_path = os.path.join(save_ckpt_dir, 'solver-model_final.pt')
        save_ckpt(save_path,
                  model=model, model_ema=model_ema,
                  optim=optimizer, args=args)

    if use_wandb and wandb:
        run.finish()


def run(train_loader, test_loader,
        config, args, device):
    # create model
    model_args = dict2namespace(config)
    model = TUnet(model_args).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        print(f'Load weights from {args.ckpt}')

    model_ema = copy.deepcopy(model).eval()
    model_ema.requires_grad_(False)

    num_params = count_params(model)
    print(f'number of parameters: {num_params}')
    config['num_params'] = num_params

    if args.distributed:
        model = DDP(model, device_ids=[device], broadcast_buffers=False)
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=config['optim']['lr'])
    scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=config['optim']['warmup'])
    # scheduler = MultiStepLR(optimizer,
    #                         milestones=config['optim']['milestone'],
    #                         gamma=0.5)
    if config['training']['loss'] == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    train(model, model_ema, train_loader,
          criterion,
          optimizer, scheduler,
          device, config, args,
          valloader=test_loader)


def subprocess_fn(rank, args):
    # setup
    if args.distributed:
        setup(rank, args.num_gpus, port=f'{args.port}')
    print(f'Running on rank: {rank}')

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    # parse configuration file
    device = rank
    batchsize = config['training']['batchsize']
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
    num_sample = config['data']['num_sample'] if 'num_sample' in config['data'] else 10000
    dir_type = config['data']['dir_type'] if 'dir_type' in config['data'] else None

    data_list = [0, 1] + list(range(3, 40))
    idx_dict = split_list(data_list, num_parts=args.num_gpus)
    # idx_dict = {
    #     0: [0],
    #     1: [10],
    #     2: [20],
    #     3: [30]
    # }
    trainset = H5Data(config['data']['datapath'],
                      config['data']['t_step'],
                      num_sample, index=idx_dict[rank], dir_type=dir_type)
    train_loader = DataLoader(trainset, batch_size=batchsize,
                              sampler=data_sampler(trainset,
                                                   shuffle=True,
                                                   distributed=False),
                              drop_last=True)
    # test set
    # testset = H5Data(config['data']['datapath'],
    #                  config['data']['t_step'],
    #                  num_sample=5000, index=[1], dir_type=dir_type)
    # test_loader = DataLoader(testset, batch_size=batchsize,
    #                          sampler=data_sampler(testset,
    #                                               shuffle=True,
    #                                               distributed=args.distributed),
    #                          drop_last=True)

    for i in range(args.repeat):
        run(train_loader, test_loader=None, config=config, args=args, device=device)
    print(f'{args.repeat} runs done!')
    if args.distributed:
        cleanup()
    print(f'Process {device} exits...')


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/cifar/tunet.yaml', help='configuration file')
    parser.add_argument('--ckpt', type=str, default=None, help='Which checkpoint to initialize the model')
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
