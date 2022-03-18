import torch
from torch.utils.data import Dataset


class myODE(Dataset):
    def __init__(self, datapath):
        super(myODE, self).__init__()
        raw = torch.load(datapath)

        self.images = torch.stack([v for v in raw.values()])

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return self.images.shape[0]

