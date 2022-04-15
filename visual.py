import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import myODE
import torch
import pandas as pd
import seaborn as sns
from mixture_sampler import Gmm_score

# datafile = torch.load('data/25gm-test-v6.pt')
# data = datafile['data'][:, -1, :].cpu().numpy()



df = pd.DataFrame(data, columns=['x', 'y'])
g = sns.jointplot(x='x', y='y', data=df, kind='kde', space=0, fill=True)

plt.show()