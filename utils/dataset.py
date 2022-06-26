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
    def __init__(self, data_dir, t_step, num_sample=10000):
        super(H5Data, self).__init__()
        self.data_dir = data_dir
        self.t_step = t_step
        if num_sample > 28000:
            raise NotImplementedError('Multi File not implemented')
        else:
            datapath = os.path.join(data_dir, 'ode_data_sd0.h5')
            f = h5py.File(datapath, 'r')
            self.dset = f['data_t33']
            self.datasize = self.dset.shape[0]

    def __getitem__(self, item):
        raw_imgs = self.dset[item]
        image = raw_imgs[::self.t_step].permute(1, 0, 2, 3)
        return torch.from_numpy(image).to(torch.float32)

    def __len__(self):
        return self.datasize