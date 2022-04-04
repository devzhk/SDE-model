from collections import OrderedDict

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
        bbox=[-2, 2, -2, 2], save_file="", xlabel="", ylabel="", cmap='Blues'):

    fig = plt.figure(figsize=(6,6))
    sns.kdeplot(data)
    # plt.scatter(mu, tau, s=4, cmap='viridis')
    # plt.xlim((-1.2, 1.2))
    # plt.ylim((-1.2, 1.2))

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()