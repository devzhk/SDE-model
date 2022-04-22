#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
import pandas as pd
import seaborn as sns

#%%
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

