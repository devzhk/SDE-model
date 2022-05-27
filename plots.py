# %%
import numpy as np
import torch
import torch.nn.functional as F
from models.fno import FNN1d
from utils.dataset import myOdeData, get_init

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# %%
config = {
    'test_datapath': 'data/25gm/25gm-test-v4-s65.pt',
    't_step': 1,
    'epsilon': 0.00001,
    'ckpt': 'exp/25gm-v4-20-m8-s65/ckpts/solver-model_final.pt',
    'layers': [20, 20, 20, 20, 20],
    'modes': [8, 8, 8, 8],
    'fc_dim': 16,
    'activation': 'leaky_relu'
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_pad = 0
# %%
testset = myOdeData(config['test_datapath'], config['t_step'])
trajectory = testset[0]

# %%
num_t = 65
t0, t1 = 1., config['epsilon']
ts = torch.linspace(t0, t1, num_t)
# %%

model = FNN1d(modes=config['modes'],
              layers=config['layers'], fc_dim=config['fc_dim'],
              in_dim=3, out_dim=2, activation=config['activation'])
# load weights
ckpt = torch.load(config['ckpt'], map_location=device)
model.load_state_dict(ckpt)

model.eval()
# put into model
with torch.no_grad():
    ini_state = trajectory[None, 0:1, :].repeat(1, num_t, 1)
    in_state = get_init(ini_state, ts)
    in_state = F.pad(in_state, (0, 0, 0, num_pad), 'constant', 0)
    pred = model(in_state)
    if num_pad > 0:
        pred = pred[:, 0:-num_pad, :]

print(pred.shape)
# %%
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')

ax.plot3D(
    ts, trajectory[:, 0].numpy(),
    trajectory[:, 1].numpy(),
    label='Ground Truth')

ax.plot3D(
    ts,
    pred[0, :, 0].numpy(),
    pred[0, :, 1].numpy(),
    label='Prediction')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('y')
plt.legend()
plt.show()

# %%
