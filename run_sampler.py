import os
import random
import numpy as np
import yaml
from tqdm import tqdm
from argparse import ArgumentParser

import torch

from models.ddpm import DDPM, Generator

from utils.ddpm_sampler import VPODE, OdeDiffusion
from utils.helper import kde, scatter


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--config', type=str, default='configs/sample/ddpm.yaml', help='configuration file')
    parser.add_argument('--log', action='store_true', help='turn on the wandb')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    #
    seed = config['seed']
    logname = config['logname']
    dataname = config['dataname']
    base_dir = f'exp/{logname}'
    save_ckpt_dir = f'{base_dir}/ckpts'
    sample_dir = f'{base_dir}/samples'
    os.makedirs(sample_dir, exist_ok=True)

    ckpt_file = config['ckpt']
    ckpt_path = os.path.join(save_ckpt_dir, ckpt_file)
    # config
    beta_min = config['beta_min']
    beta_max = config['beta_max']
    batchsize = config['batchsize']
    num_batch = config['num_samples'] // batchsize
    scale_sigma = config['scale'] if 'scale' in config else False
    # add device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # set random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    model = DDPM(config).to(device)
    # model = Generator().to(device)
    # ckpt = torch.load('exp/ref_ddpm_25gm.pt', map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    print(scale_sigma)
    vpode = VPODE(model, beta_min, beta_max, scale_sigma=scale_sigma)
    ode_diff = OdeDiffusion(config, vpode, img_shape=[2], device=device)

    images_list = []
    for i in range(num_batch):
        images_batch = ode_diff.uncond_sample(batchsize, bs_id=-1)
        images_list.append(images_batch)
    data = torch.cat(images_list, dim=0).detach().cpu()

    xT_fig_name = os.path.join(sample_dir, f'xT_{dataname}_{seed}.png')
    x0_fig_name = os.path.join(sample_dir, f'x0_{dataname}_{seed}.png')
    kde(data[:, 0, :], xT_fig_name, dim=2)
    kde(data[:, -1, :], x0_fig_name, dim=2)
    # scatter(data[:, -1, :].detach().cpu().numpy(), x0_fig_name)
    torch.save(
        {
            'data': data,
        },
        f'{sample_dir}/{dataname}.pt'
    )