import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN1d


class myODE(Dataset):
    def __init__(self, datapath):
        super(myODE, self).__init__()
        raw = torch.load(datapath)

        images = torch.stack([v for v in raw.values()])
        images = images.reshape(images.shape[0], images.shape[1], -1)
        self.images = images.permute(1, 0, 2)
        # batchsize x T x 3072

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return self.images.shape[0]


def run(configs=None):
    dim = 3 * 32 * 32
    batchsize = 64
    num_epoch = 2000
    base_dir = 'exp/seed1234/'
    save_img_dir = f'{base_dir}/figs'
    os.makedirs(save_img_dir, exist_ok=True)

    save_ckpt_dir = f'{base_dir}/ckpts'
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # construct dataset
    dataset = myODE('data/ode_data_sd1.pt')

    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

    # define operator for solving SDE
    layers = [128, 128, 128, 128]
    modes1 = [8, 8, 8]
    fc_dim = 1024
    activation = 'gelu'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FNN1d(modes=modes1,
                  fc_dim=fc_dim,
                  layers=layers,
                  activation=activation,
                  in_dim=dim, out_dim=dim).to(device)
    # define optimizer and criterion
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[500, 800, 1200, 1600], gamma=0.5)
    # train
    # hyperparameter

    model.train()

    criterion = nn.MSELoss()
    pbar = tqdm(list(range(num_epoch)), dynamic_ncols=True)

    for e in pbar:
        epoch_loss = 0
        for img in train_loader:
            model.zero_grad()

            img = img.to(device)
            init = img[:, 0:1, :].repeat(1, img.shape[1], 1)
            pred = model(init)

            loss = criterion(pred, img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= len(train_loader)
        pbar.set_description(
            (
                f'Epoch: {e}; avg loss: {epoch_loss}'
            )
        )

        if e % 50 == 0:
            image = img[:, -1, :].reshape(batchsize, 3, 32, 32)
            model_pred = pred[:, -1, :].reshape(batchsize, 3, 32, 32)
            save_image((image + 1) * 0.5, f'{save_img_dir}/train_{e}_sample.png', nrow=int(np.sqrt(image.shape[0])))
            save_image((model_pred + 1) * 0.5, f'{save_img_dir}/train_{e}_pred.png', nrow=int(np.sqrt(pred.shape[0])))
            torch.save(model.state_dict(), f'{save_ckpt_dir}/solver-model_{e}.pt')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    run()






