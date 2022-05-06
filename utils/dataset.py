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

