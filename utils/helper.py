from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from .sde_lib import VPSDE, VESDE, subVPSDE


def count_params(model):
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


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
        plt.cla()
        plt.close(fig)
    else:
        plt.show()


def group_kde(data, labels,
              save_file=None):
    fig = plt.figure(figsize=(6, 6))

    for ys, label in zip(data, labels):
        sns.kdeplot(ys, label=label)

    plt.legend()
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
        plt.cla()
        plt.close()
    else:
        plt.show()


'''
helper functions from SDE
'''


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.
    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))