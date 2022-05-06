import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN2d
from utils.optim import Adam


class myOdeData(Dataset):
    def __init__(self, datapath):
        super(myOdeData, self).__init__()
        raw = torch.load(datapath)
        self.codes = raw['code'].detach().clone()
        self.images = raw['image'].detach().clone()

    def __getitem__(self, idx):
        return self.codes[idx], self.images[idx]

    def __len__(self):
        return self.codes.shape[0]


def train(model, dataloader,
          criterion,
          optimizer, scheduler,
          device, config):

    t_dim = config['t_dim']
    t_step = config['t_step']
    num_t = math.ceil(t_dim / t_step)
    logname = config['logname']
    save_step = config['save_step']
    # prepare log dir
    base_dir = f'exp/{logname}/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)
    model.train()
    pbar = tqdm(list(range(config['num_epoch'])), dynamic_ncols=True)
    for e in pbar:
        train_loss = 0
        for states in dataloader:
            ini_state = states[:, 0:1, :].repeat(1, num_t, 1)
    # TODO: complete the generic function
    pass


seed = 32123
batchsize = 256
# construct dataset
# dataset = myOdeData('data/data.pt')
base_dir = f'exp/cifar10-seed{seed}/'
# dataset = myOdeData(f'data/ode_data_sd1.pt')
dataset = myOdeData(f'data/test_data_50k.pt')
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

# define operator for solving SDE
layers = [64, 64, 64, 64, 64, 64, 64]
modes1 = [16, 16, 16, 16, 16, 16]
modes2 = [16, 16, 16, 16, 16, 16]
fc_dim = 64
activation = 'gelu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = FNN2d(modes1=modes1, modes2=modes1,
              fc_dim=fc_dim, layers=layers,
              in_dim=3, out_dim=3,
              activation=activation).to(device)
# define optimizer and criterion
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=0.5)
criterion = nn.MSELoss()
# train
# hyperparameter
num_epoch = 500
model.train()

pbar = tqdm(list(range(num_epoch)), dynamic_ncols=True)

save_img_dir = f'{base_dir}/figs'
os.makedirs(save_img_dir, exist_ok=True)

save_ckpt_dir = f'{base_dir}/ckpts'
os.makedirs(save_ckpt_dir, exist_ok=True)

for e in pbar:
    train_loss = 0
    for code, image in train_loader:
        code = code.to(device)
        image = image.to(dtype=torch.float32, device=device)
        pred = model(code)
        loss = criterion(pred, image)
        # update model
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    train_loss /= len(train_loader)
    pbar.set_description(
        (
            f'Epoch :{e}, Loss: {train_loss}'
        )
    )

    if e % 50 == 0:
        # samples = pred.clamp(0.0, 1.0)
        if batchsize > 64:
            image, pred = image[:64], pred[:64]  # first 100
        save_image((image + 1) * 0.5, f'{save_img_dir}/train_{e}_sample.png', nrow=int(np.sqrt(image.shape[0])))
        save_image((pred + 1) * 0.5, f'{save_img_dir}/train_{e}_pred.png', nrow=int(np.sqrt(pred.shape[0])))
        torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')

torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')
