import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN2d


class myMNIST(Dataset):
    def __init__(self, datapath):
        super(myMNIST, self).__init__()
        raw = torch.load(datapath)
        self.codes = raw['code']
        self.images = raw['image']

    def __getitem__(self, idx):
        return self.codes[idx], self.images[idx]

    def __len__(self):
        return self.codes.shape[0]


batchsize = 100
# construct dataset
dataset = myMNIST('data/data.pt')
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

# define operator for solving SDE
layers = [32, 32, 32]
modes1 = [12, 12]
modes2 = [12, 12]
fc_dim = 64
activation = 'gelu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = FNN2d(modes1=modes1, modes2=modes1,
              fc_dim=fc_dim, layers=layers,
              activation=activation, in_dim=1).to(device)
# define optimizer and criterion
optimizer = Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()
# train
# hyperparameter
num_epoch = 500
model.train()

pbar = tqdm(list(range(num_epoch)), dynamic_ncols=True)

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

    train_loss /= len(train_loader)
    pbar.set_description(
        (
            f'Epoch :{e}, Loss: {train_loss}'
        )
    )
    samples = pred.clamp(0.0, 1.0)
    save_image(samples, f'figs/train_{e}.png', nrow=int(np.sqrt(batchsize)))



torch.save(model.state_dict(), 'ckpts/solver-model.pt')






