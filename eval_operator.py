import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from torchvision.utils import save_image

from models.fno import FNN2d


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

batchsize = 100
# construct dataset
base_dir = 'exp/seed1234'
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

save_ckpt_dir = f'{base_dir}/ckpts'
model_fn = f'{save_ckpt_dir}/solver-model_final.pt'
model = FNN2d(modes1=modes1, modes2=modes1,
              fc_dim=fc_dim, layers=layers,
              in_dim=3, out_dim=3,
              activation=activation).to(device)
model.load_state_dict(torch.load(model_fn))

save_img_dir = f'{base_dir}/evals'
os.makedirs(save_img_dir, exist_ok=True)

for i, (code, image) in enumerate(train_loader):
    code = code.to(device)
    image = image.to(dtype=torch.float32, device=device)
    pred = model(code)

    if i < 10:  # 10 batches
        # code = code.clamp(-1.0, 1.0)
        # image = image.clamp(-1.0, 1.0)
        # pred = pred.clamp(-1.0, 1.0)
        save_image((code + 1) * 0.5, f'{save_img_dir}/init_{i}.png', nrow=int(np.sqrt(batchsize)))
        save_image((image + 1) * 0.5, f'{save_img_dir}/image_{i}.png', nrow=int(np.sqrt(batchsize)))
        save_image((pred + 1) * 0.5, f'{save_img_dir}/pred_{i}.png', nrow=int(np.sqrt(batchsize)))

    else:
        break
