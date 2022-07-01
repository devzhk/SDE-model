import os
import math
import yaml
import torch
from argparse import ArgumentParser
from cleanfid import fid

from models.tunet import TUnet
from utils.helper import dict2namespace



def compute_fid(generator,
                t0, t1,
                num_t,
                z_dim=3072,
                img_size=32,
                dataname='cifar10',
                datasplit='train',
                device=torch.device('cpu')):
    generator.eval()
    timesteps = torch.linspace(t0, t1, num_t, device=device)
    def gen(z):

        img = generator(z.reshape(-1, 3, img_size, img_size), timesteps)[:, :, -1]
        img_int = img.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8)
        return img_int
    # gen = lambda z: generator(z.reshape(-1, 3, img_size, img_size)).add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8)

    score = fid.compute_fid(gen=gen, dataset_name=dataname, dataset_split=datasplit,
                            dataset_res=img_size, z_dim=z_dim)
    print(f'FID score: {score}')
    return score


if __name__ == '__main__':
    parser = ArgumentParser('basic parser for evaluating FID score')
    parser.add_argument('--dataname', type=str, default='cifar10')
    parser.add_argument('--datasplit', type=str, default='train')
    parser.add_argument('--config', type=str, default='configs/cifar/tunet-kd.yaml')
    parser.add_argument('--ckpt', type=str, default='exp/cifar-tunet-112k/ckpts/solver-model_20000.pt')
    parser.add_argument('--logdir', type=str, default='log/default')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=3072)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse configuration
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    model_args = dict2namespace(config)

    t_dim = model_args.data.t_dim
    t_step = model_args.data.t_step
    num_t = math.ceil(t_dim / t_step)

    # create model from configuration
    generator = TUnet(model_args).to(device)
    # Load weights
    print(f'Load weights from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    generator.load_state_dict(ckpt['ema'])

    compute_fid(generator,
                t0=1.0, t1=model_args.data.epsilon,
                num_t=num_t,
                z_dim=args.z_dim,
                img_size=args.img_size,
                dataname=args.dataname,
                datasplit=args.datasplit,
                device=device)