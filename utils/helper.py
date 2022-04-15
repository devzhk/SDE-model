from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def remove_module(ckpt):
    state_dict = OrderedDict()
    for key, value in ckpt.items():
        name = key.replace('module.', '')
        state_dict[name] = value
    return state_dict


# calculate gaussian kde estimate for a dataset
def kde(data, # 1d array
        save_file="",
        dim=1):

    fig = plt.figure(figsize=(6,6))
    if dim == 1:
        sns.kdeplot(data)
    elif dim == 2:
        df = pd.DataFrame(data.cpu().detach().numpy(), columns=['x', 'y'])
        g = sns.jointplot(x='x', y='y', data=df, kind='kde', space=0, fill=True)
    # plt.scatter(mu, tau, s=4, cmap='viridis')
    # plt.xlim((-1.2, 1.2))
    # plt.ylim((-1.2, 1.2))

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def group_kde(data, labels,
              save_file=''):
    fig = plt.figure(figsize=(6, 6))

    for ys, label in zip(data, labels):
        sns.kdeplot(ys, label=label)

    plt.legend()
    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()