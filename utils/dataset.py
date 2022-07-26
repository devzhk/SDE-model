import os
import psutil
import h5py

import torch
from torch.utils.data import Dataset


def split_list(arr, num_parts=4):
    data_dict = {}
    chunk_size = len(arr) // num_parts
    for i in range(num_parts):
        data_dict[i] = arr[i * chunk_size: (i + 1) * chunk_size]
    rem = len(arr) % num_parts
    for j in range(rem):
        data_dict[j].append(arr[num_parts * chunk_size + j])
    return data_dict


def get_init(x, ts):
    '''
    x: (batchsize, t_dim, 1)
    ts: (t_dim, )
    '''
    t_data = ts.repeat(x.shape[0], 1).unsqueeze(-1)
    return torch.cat([x, t_data], dim=-1)


class myOdeData(Dataset):
    def __init__(self, datapath, t_step, num_sample=None):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        data = raw['data'].detach().clone()
        if num_sample is None:
            self.data = data[:, 0::t_step]
        else:
            self.data = data[:num_sample, 0::t_step]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class ImageData(Dataset):
    def __init__(self, data_dir, t_step, index=[0,], dir_type=None):
        super(ImageData, self).__init__()
        self.data_dir = data_dir
        self.t_step = t_step
        self.index = index
        self.dir_type = dir_type
        if dir_type == 'subfolder':
            datapath = os.path.join(data_dir, f'seed{index[0]}', f'ode_data_sd{index[0]}.h5')
        else:
            datapath = os.path.join(data_dir, f'ode_data_sd{index[0]}.h5')
        with h5py.File(datapath, 'r') as f:
            num_per_file = len(f['data_t33'])
        self.num_sample = len(index) * num_per_file
        self.num_per_file = num_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.num_per_file
        data_idx = idx % self.num_per_file
        if self.dir_type == 'subfolder':
            datapath = os.path.join(self.data_dir,
                                    f'seed{self.index[file_idx]}',
                                    f'ode_data_sd{self.index[file_idx]}.h5')
        else:
            datapath = os.path.join(self.data_dir,
                                    f'ode_data_sd{self.index[file_idx]}.h5')
        with h5py.File(datapath, 'r') as f:
            data = f['data_t33'][data_idx]
        return data

    def __len__(self):
        return self.num_sample


class H5Data(Dataset):
    def __init__(self, data_dir, t_step, num_sample=10000, index=[0,], dir_type=None):
        super(H5Data, self).__init__()
        self.data_dir = data_dir
        self.t_step = t_step
        if num_sample > 28000:
            trunk_list = []
            for idx in index:
                if dir_type == 'subfolder':
                    datapath = os.path.join(data_dir, f'seed{idx}', f'ode_data_sd{idx}.h5')
                else:
                    datapath = os.path.join(data_dir, f'ode_data_sd{idx}.h5')
                with h5py.File(datapath, 'r') as f:
                    dset = f['data_t33'][:, ::self.t_step]
                # memory_use = psutil.Process().memory_info().rss / (1024 * 1024)
                # print(f'Memory usage: {memory_use} MB')
                print(f'Read data from {datapath}')
                dset = torch.from_numpy(dset).to(torch.float32)
                trunk_list.append(dset)
                # memory_use = psutil.Process().memory_info().rss / (1024 * 1024)
                # print(f'Memory usage: {memory_use} MB')
            self.dset = torch.cat(trunk_list, dim=0).permute(0, 2, 1, 3, 4)
            self.datasize = self.dset.shape[0]
        else:
            idx = index[0]
            datapath = os.path.join(data_dir, f'ode_data_sd{idx}.h5')
            with h5py.File(datapath, 'r') as f:
                dset = f['data_t33'][0:num_sample, ::self.t_step]
            self.dset = torch.from_numpy(dset).to(torch.float32).permute(0, 2, 1, 3, 4)
            self.datasize = num_sample
        self.curr_idx = 0

    def __getitem__(self, item):
        # B, C, T, H, W
        img = self.dset[item]
        return img

    def __len__(self):
        return self.datasize

    def get_batch(self, B):
        batch = self.dset[self.curr_idx: self.curr_idx + B, :, -1]
        self.curr_idx += B
        return batch