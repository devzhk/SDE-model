import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN1d

from utils.helper import kde, group_kde


def get_init(x, ts):
    '''
    x: (batchsize, t_dim, 1)
    ts: (t_dim, )
    '''
    t_data = ts.repeat(x.shape[0], 1).unsqueeze(-1)
    return torch.cat([x, t_data], dim=-1)


class myOdeData(Dataset):
    def __init__(self, datapath):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        self.data = raw['data'].detach().clone()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


batchsize = 1000
t_dim = 50
# construct dataset
# dataset = myOdeData('data/data.pt')
base_dir = 'exp/1dGM_seed1234/'
dataset = myOdeData(f'data/1dgm-test.pt')
test_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

# define operator for solving SDE
layers = [2, 2, 2]
modes1 = [6, 6]
modes2 = [6, 6]
fc_dim = 2
activation = 'gelu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = FNN1d(modes=modes1,
              fc_dim=fc_dim, layers=layers,
              in_dim=2, out_dim=1,
              activation=activation).to(device)
# define optimizer and criterion
ckpt = torch.load('exp/1dGM_seed1234/ckpts/solver-model_final.pt')
model.load_state_dict(ckpt)
criterion = nn.MSELoss()
# train
# hyperparameter
epsilon = 1e-5
t0, t1 = 1., epsilon
ts = torch.linspace(t0, t1, t_dim)

save_img_dir = f'{base_dir}/figs'
os.makedirs(save_img_dir, exist_ok=True)

save_ckpt_dir = f'{base_dir}/ckpts'
os.makedirs(save_ckpt_dir, exist_ok=True)

model.eval()
test_err = 0
for states in test_loader:
    in_state = get_init(states, ts)

    pred = model(in_state)
    loss = criterion(pred, states)
    # update model
    test_err += loss.item()

print(f'Test Mean squared error: {test_err}')


zip_state = [pred[:, -1, 0].detach().numpy(), states[:, -1, 0].detach().numpy()]
labels = ['Prediction', 'Truth']
group_kde(zip_state, labels, f'figs/1dGM/test.png')

