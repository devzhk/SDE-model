from argparse import ArgumentParser
import yaml

import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from models.layers import ResnetBlockDDPM, BBlock, Upsample, Downsample, TimeConv, get_timestep_embedding
from models.tunet import TUnet
from utils.distributed import setup, cleanup
from utils.helper import dict2namespace, count_params


# Test for ResnetBlockDDPM
def test_resblockDDPM(device, args):
    act = nn.SiLU()
    net = ResnetBlockDDPM(act=act, in_ch=64, out_ch=32, temb_dim=3, conv_shortcut=True)
    B, C, T, H, W = 4, 64, 4, 32, 32
    image = torch.randn((B, C, T, H, W))
    time_emb = torch.randn((T, 3))
    pred = net(image, time_emb)
    print(pred.shape)


# Test for
def test_bblock(device, args):
    block = BBlock(in_ch=3, out_ch=32)
    B, C, T, H, W = 4, 3, 4, 32, 32
    image = torch.randn((B, C, H, W))
    time_emb = torch.randn((T, 32))
    pred = block(image, time_emb)
    print(pred.shape)


# test upsample or downsample
def test_sample(device, args):
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
    # setup
    if args.distributed:
        setup(rank, args.num_gpus, port='8902')
    print(f'Running on rank: {rank}')
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    B, C, T, H, W = 4, 4, 5, 8, 8
    xs = torch.randn((B, C, T, H, W), device=rank)
    if args.distributed:
        ys = xs[rank * 2: (rank+1) * 2]
    else:
        ys = xs

    act = nn.SiLU()
    block = TimeConv(in_ch=C, out_ch=C, modes=2, act=act).to(rank)
    if args.distributed:
        block = DDP(block, device_ids=[rank], broadcast_buffers=False)
        model = block.module
    else:
        model = block


    # print(f'Input {ys[0, 1, 2]}')
    pred = block(ys)
    loss = torch.mean(pred)
    loss.backward()

    print(model.t_conv.weights1.grad[0, 1])
    print(model.nin.W.grad[0, 1])
    print(model.nin.b.grad[0])
    # print(f'Output {pred[0, 1, 2]}')
    if args.distributed:
        cleanup()


def test_tunet(rank, args):
    with open(args.config, 'r') as f:
        model_config = yaml.load(f, yaml.FullLoader)
    model_args = dict2namespace(model_config)
    model = TUnet(model_args)
    num_params = count_params(model)
    print(f'Number of model parameters: {num_params}')
    B, C, T, H, W = 4, 3, 7, 32, 32
    timesteps = torch.linspace(0, 1, T)

    image = torch.randn((B, C, H, W))
    pred = model(image, timesteps)
    print(pred.shape)


def test_temb(device, args):
    timesteps = torch.linspace(0, 1, 10)
    temb = get_timestep_embedding(timesteps, 128)
    print(temb.shape)


if __name__ == '__main__':

    def split_list(arr, num_parts=4):
        data_dict = {}
        chunk_size = len(arr) // num_parts
        for i in range(num_parts):
            data_dict[i] = arr[i * chunk_size: (i + 1) * chunk_size]
        rem = len(arr) % num_parts
        for j in range(rem):
            data_dict[j].append(arr[num_parts * chunk_size + j])
        return data_dict

    arr = [0, 1] + list(range(3, 40))
    data_dict = split_list(arr, 8)
    print(data_dict)

    # parser = ArgumentParser(description='Basic parser')
    # parser.add_argument('--num_gpus', type=int, default=1)
    # parser.add_argument('--config', type=str, default='configs/cifar/tunet-kd.yaml')
    # args = parser.parse_args()
    # args.distributed = args.num_gpus > 1
    #
    # # subprocess = test_fno
    # # subprocess = test_temb
    # subprocess = test_tunet
    #
    # device = 0 if torch.cuda.is_available() else 'cpu'
    # if args.distributed:
    #     mp.spawn(subprocess, args=(args, ), nprocs=args.num_gpus)
    # else:
    #     subprocess(device, args)
    # test_bblock()
    # test_sample()

