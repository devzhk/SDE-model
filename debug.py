from tqdm import tqdm

import h5py
import numpy as np

import torch

datapath = 'data/cifar10/ode_data_test_t9.pt'
outpath = 'data/cifar10/data_test_t9.h5'

data = torch.load(datapath).numpy()
N, T, C, H, W = data.shape
print(data.shape)
# chunk_size = 1000
# num_chunks = N // chunk_size
#
# with h5py.File(outpath, 'w') as f:
#     dset = f.create_dataset('test_t9', (chunk_size, T, C, H, W), maxshape=(None, T, C, H, W))
#     dset[:] = data[0:chunk_size]
#     for i in range(num_chunks - 1):
#         dset.resize(size=(2 + i) * chunk_size, axis=0)
#         dset[-chunk_size:] = data[(i+1) * chunk_size: (i+2) * chunk_size]
#         print(dset.shape)
#
# print('Done!')
