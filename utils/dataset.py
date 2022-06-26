import os
import h5py

import torch
from torch.utils.data import Dataset


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
    def __init__(self, datapath, t_step, num_sample=None):
        super(ImageData, self).__init__()
        raw_data = torch.load(datapath)
        # N, T, C, H, W
        if num_sample is None:
            self.data = raw_data[:, 0::t_step]
        else:
            self.data = raw_data[:num_sample, 0::t_step]
        self.num_sample = self.data.shape[0]
        # N, C, T, H, W

    def __getitem__(self, idx):
        return self.data[idx].permute(1, 0, 2, 3)

    def __len__(self):
        return self.num_sample


class H5Data(Dataset):
    def __init__(self, data_dir, t_step, num_sample=10000, index=[0,]):
        super(H5Data, self).__init__()
        self.data_dir = data_dir
        self.t_step = t_step
        if num_sample > 28000:
            trunk_list = []
            for idx in index:
                datapath = os.path.join(data_dir, f'ode_data_sd{idx}.h5')
                with h5py.File(datapath, 'r') as f:
                    dset = f['data_t33'][:, ::self.t_step]
                dset = torch.from_numpy(dset).to(torch.float32)
                trunk_list.append(dset)
            self.dset = torch.cat(trunk_list, dim=0).permute(0, 2, 1, 3, 4)
            self.datasize = self.dset.shape[0]
        else:
            idx = index[0]
            datapath = os.path.join(data_dir, f'ode_data_sd{idx}.h5')
            with h5py.File(datapath, 'r') as f:
                dset = f['data_t33'][0:num_sample, ::self.t_step]
            self.dset = torch.from_numpy(dset).to(torch.float32).permute(0, 2, 1, 3, 4)
            self.datasize = num_sample

    def __getitem__(self, item):
        img = self.dset[item]
        # tensor_imgs = torch.from_numpy(raw_imgs[::self.t_step]).to(torch.float32).permute(1, 0, 2, 3)
        return img

    def __len__(self):
        return self.datasize