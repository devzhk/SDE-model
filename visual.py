# %%
from utils.helper import scatter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mixture_sampler import Gmm_score, VPODE
import torch

# %%
data = torch.load('exp/cifar_sample/images/seed1234/ode_data_sd1234.pt', map_location='cpu')
print(data.shape)


# gmm = Gmm_score(dim=2, mu=2.0, sigma2=1e-4, num_modes=25)
# ode = VPODE(gmm)
#
# # %%
# ts = torch.linspace(0, 1, 10)
# xt = torch.randn(3, 10, 2)
# # %%
# that = torch.fft.rfft
#
# # %%
# dataset = []
# for i in range(10000 // 25):
#     for x in range(-2, 3):
#         for y in range(-2, 3):
#             point = np.random.randn(2) * 0.05
#             point[0] += 2 * x
#             point[1] += 2 * y
#             dataset.append(point)
# dataset = np.array(dataset, dtype='float32')
# np.random.shuffle(dataset)
# dataset /= 2.828  # stdev
#
# scatter(dataset, 'figs/ref_data.png')

# %%
# datafile = torch.load('data/25gm-test-v6.pt')
# data = datafile['data'][:, -1, :].cpu().numpy()
# mus = [[2.0, 2.0],
#        [2.0, -2.0],
#        [-2.0, 2.0],
#        [-2.0, -2.0]]
# cov = np.eye(2) * 0.01
#
# data_list = []
# for mu in mus:
#     pts = np.random.multivariate_normal(mu, cov, size=1000)
#     data_list.append(pts)
# data = np.concatenate(data_list, axis=0)
#
# df = pd.DataFrame(data, columns=['x', 'y'])
# g = sns.jointplot(x='x', y='y', data=df, kind='kde', space=0, fill=True)
#
# plt.show()
# # %%
# mus = [[2.0, 2.0],
#        [2.0, -2.0],
#        [-2.0, 2.0],
#        [-2.0, -2.0]]
# cov = np.eye(2) * 0.0001
#
# data_list = []
# for mu in mus:
#     pts = np.random.multivariate_normal(mu, cov, size=1000)
#     data_list.append(pts)
# data = np.concatenate(data_list, axis=0)
#
# df = pd.DataFrame(data, columns=['x', 'y'])
# g = sns.jointplot(x='x', y='y', data=df, kind='kde', space=0, fill=True)
#
# plt.show()
# # %%
# mus = [[2.0, 2.0],
#        [2.0, -2.0],
#        [-2.0, 2.0],
#        [-2.0, -2.0]]
# cov = np.eye(2) * 0.000001
#
# data_list = []
# for mu in mus:
#     pts = np.random.multivariate_normal(mu, cov, size=1000)
#     data_list.append(pts)
# data = np.concatenate(data_list, axis=0)
#
# df = pd.DataFrame(data, columns=['x', 'y'])
# g = sns.jointplot(x='x', y='y', data=df, kind='kde', space=0, fill=True)
#
# plt.show()
# %%
