import torch

from models.unet3d import Unet3D

B = 8
C = 3
T = 16
H, W = 32, 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = torch.randn((B, C, T, H, W), device=device)

model = Unet3D(dim=32).to(device)

pred = model(data, 0.0)

print(pred.shape)



