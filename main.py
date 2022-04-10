import os
import random
from argparse import ArgumentParser

import numpy as np

import torch

from utils.helper import kde
from mixture_sampler import OdeDiffusion

# def sample_normal(mu, sigma):
#     sample = np.random.normal(mu, sigma)
#     return sample
#
# num_sample = 3000
#
# mus = np.linspace(-1, 1, num=3)
# sigma = 0.1
#
#
# idx = np.random.randint(0, 3, size=num_sample)
#
# data = [sample_normal(mus[i], sigma) for i in idx]
#
# kde(data, save_file='figs/test_gmm.png')

if __name__ == '__main__':
    parser = ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--step_size', type=float, default=1e-4, help='step size for ODE Euler method')
    parser.add_argument('--seed', type=int, default=500, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=5000, help='batch size')
    parser.add_argument('--t_size', type=int, default=50, help='number of states in ode to save')
    parser.add_argument('--save_dir', type=str, default='save/exp_loss0_randt')
    args = parser.parse_args()

    log_dir = os.path.join(args.save_dir, f'sample_x_seed{str(args.seed)}_ss{args.step_size}')
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    # add device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    ode_diffusion = OdeDiffusion(args, device=device).to(device)
    images_list = ode_diffusion.uncond_sample(args.batch_size, bs_id=-1)
    data = torch.stack(images_list, dim=1)
    bw = 0.1
    kde(data[:, 0, 0], f'figs/1dGM/xT_train_{bw}_{args.seed}.png', bw)
    kde(data[:, -1, 0], f'figs/1dGM/x0_train_{bw}_{args.seed}.png', bw)
    torch.save(
        {
            'data': data
        },
        'data/1dgm-train.pt'
    )
    print('done')