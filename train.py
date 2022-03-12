import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN2d
import utils


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


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batchsize = 400
# construct dataset
# dataset = myOdeData('data/data.pt')
base_dir = 'exp/seed1234/'
dataset = myOdeData(f'data/test_data.pt')
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

# define operator for solving SDE
layers = [32, 32, 32]
modes1 = [12, 12]
modes2 = [12, 12]
fc_dim = 64
activation = 'gelu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

model = FNN2d(modes1=modes1, modes2=modes1,
              fc_dim=fc_dim, layers=layers,
              in_dim=3, out_dim=3,
              activation=activation).to(device)
# define optimizer and criterion
optimizer = Adam(model.parameters(), lr=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[300, 450, 600, 750], gamma=0.5)
criterion = nn.MSELoss()
# train
# hyperparameter
num_epoch = 1000
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

    if e % 10 == 0:
        # samples = pred.clamp(0.0, 1.0)
        if batchsize > 100:
            image, pred = image[:100], pred[:100]  # first 100
        save_image((image + 1) * 0.5, f'{save_img_dir}/train_{e}_sample.png', nrow=int(np.sqrt(image.shape[0])))
        save_image((pred + 1) * 0.5, f'{save_img_dir}/train_{e}_pred.png', nrow=int(np.sqrt(pred.shape[0])))

    if e % 100 == 0:
        torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')

torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_final.pt')
