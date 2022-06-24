from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import math
from models.layers import ResnetBlockDDPM, BBlock, Upsample, Downsample, TimeConv
from utils.distributed import setup, cleanup



# Test for ResnetBlockDDPM
def test_resblockDDPM():
    act = nn.SiLU()
    net = ResnetBlockDDPM(act=act, in_ch=64, out_ch=32, temb_dim=3, conv_shortcut=True)
    B, C, T, H, W = 4, 64, 4, 32, 32
    image = torch.randn((B, C, T, H, W))
    time_emb = torch.randn((T, 3))
    pred = net(image, time_emb)
    print(pred.shape)


# Test for
def test_bblock():
    block = BBlock(in_ch=3, out_ch=32)
    B, C, T, H, W = 4, 3, 4, 32, 32
    image = torch.randn((B, C, H, W))
    time_emb = torch.randn((T, 32))
    pred = block(image, time_emb)
    print(pred.shape)


# test upsample or downsample
def test_sample():
    B, C, T, H, W = 4, 16, 5, 8, 8
    image = torch.randn((B, C, T, H, W))

    ublock = Upsample(C, True)
    pred_up = ublock(image)
    print(f'Upsampled data shape: {pred_up.shape}')

    dblock = Downsample(C, True)
    pred_down = dblock(image)
    print(f'Downsampled data shape: {pred_down.shape}')


# test FNO block
def test_fno(rank, args):
    if args.distributed:




if __name__ == '__main__':
    parser = ArgumentParser(description='Basic parser')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    subprocess = test_fno

    if args.distributed:
        mp.spawn(subprocess, args=(args, ), nprocs=args.num_gpus)
    else:
        subprocess(0, args)
    # test_bblock()
    # test_sample()

