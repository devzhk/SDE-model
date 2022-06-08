from argparse import ArgumentParser
import torch


def convert(args):
    datapath = args.path
    t_dim = 65
    t_step = args.t_step
    num_t = (t_dim - 1) // t_step
    raw_data = torch.load(datapath)
    print(f'original data shape: {raw_data.shape}')
    sub_data = raw_data[:, 0::t_step].clone()
    print(f'subsampled data shape: {sub_data.shape}')
    save_path = datapath.replace('.pt', f'_t{num_t + 1}.pt')
    torch.save(sub_data, save_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for converting data format')
    parser.add_argument('--path', type=str)
    parser.add_argument('--t_step', type=int)
    args = parser.parse_args()
    convert(args)