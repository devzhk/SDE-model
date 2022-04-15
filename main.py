import os
import random
from argparse import ArgumentParser

import numpy as np

import torch

from utils.helper import kde
from mixture_sampler import OdeDiffusion, Gmm_score

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
    parser.add_argument('--seed', type=int, default=551, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--t_size', type=int, default=50, help='number of states in ode to save')
    parser.add_argument('--dataname', type=str, default='25gm-train', help='name of the data file')
    parser.add_argument('--save_dir', type=str, default='save/exp_loss0_randt')
    parser.add_argument('--dimension', type=int, default=2, help='dimension of data')
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

    gmm_model = Gmm_score(dim=args.dimension, mu=2.0, num_modes=25)
    ode_diffusion = OdeDiffusion(args, gmm_score=gmm_model, img_shape=[args.dimension], device=device).to(device)
    num_batch = args.num_samples // args.batch_size
    images_list = []
    for i in range(num_batch):
        images_batch = ode_diffusion.uncond_sample(args.batch_size, bs_id=-1)
        images_list.append(images_batch)
    data = torch.cat(images_list, dim=0)

    xT_fig_name = f'figs/{args.dimension}DGM/xT_{args.dataname}_{args.seed}.png'
    x0_fig_name = f'figs/{args.dimension}DGM/x0_{args.dataname}_{args.seed}.png'
    if args.dimension == 1:
        kde(data[:, 0, 0], xT_fig_name, dim=args.dimension)
        kde(data[:, -1, 0], x0_fig_name, dim=args.dimension)
    elif args.dimension == 2:
        kde(data[:, 0, :], xT_fig_name, dim=args.dimension)
        kde(data[:, -1, :], x0_fig_name, dim=args.dimension)
    torch.save(
        {
            'data': data
        },
        f'data/{args.dimension}Dgm-{args.dataname}.pt'
    )
    print('done')