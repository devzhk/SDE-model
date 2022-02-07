import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision.utils import save_image

from tqdm import tqdm

from models.fno import FNN2d

'''
TODO: 
load from a test set. Sample image using trained operator. 
'''