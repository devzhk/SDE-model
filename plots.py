# %%
import yaml
import torch
from models.tunet import TUnet
from utils.helper import dict2namespace, count_params
# %%
device = torch.device('cpu')
ckpt = torch.load('ckpts/checkpoint_14.pth', map_location=device)

# %%
config_path = 'configs/cifar/tunet-kd.yaml'
with open(config_path, 'r') as f:
    model_config = yaml.load(f, yaml.FullLoader)
    model_config = dict2namespace(model_config)

model = TUnet(model_config)
# %%
for name, param in model.named_parameters():
    print(name)
    print(param)
    break
# %%
