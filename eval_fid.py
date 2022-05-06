import os
import torch
from argparse import ArgumentParser
from cleanfid import fid

from models.fno import FNN2d


def compute_fid(generator, args):
    print(f'Load weights from {args.ckpt}')
    ckpt = torch.load(args.ckpt)
    generator.load_state_dict(ckpt)
    img_size = args.img_size

    gen = lambda z: generator(z.reshape(-1, 3, img_size, img_size)).add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8)

    score = fid.compute_fid(gen=gen, dataset_name=args.dataname, dataset_split=args.datasplit,
                            dataset_res=img_size, z_dim=args.z_dim)
    print(f'FID score: {score}')
    return score


if __name__ == '__main__':
    parser = ArgumentParser('basic parser for evaluating FID score')
    parser.add_argument('--dataname', type=str, default='cifar10')
    parser.add_argument('--datasplit', type=str, default='train')
    parser.add_argument('--ckpt', type=str, default='exp/cifar10-seed32123/ckpts/solver-model_final.pt')
    parser.add_argument('--logdir', type=str, default='log/default')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=3072)
    args = parser.parse_args()

    layers = [64, 64, 64, 64, 64, 64, 64]
    modes1 = [16, 16, 16, 16, 16, 16]
    modes2 = [16, 16, 16, 16, 16, 16]
    fc_dim = 64
    activation = 'gelu'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator = FNN2d(modes1=modes1, modes2=modes1,
                      fc_dim=fc_dim, layers=layers,
                      in_dim=3, out_dim=3,
                      activation=activation).to(device)
    compute_fid(generator, args)