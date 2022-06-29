import sys
import argparse
from typing import Any

from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import jax
# from .sde_lib import VPSDE, VESDE, subVPSDE



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
        if dim == 2:
            plt.close(g.fig)
    else:
        plt.show()


def scatter(data, save_file=None):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(6,6))
    plt.scatter(data[:, 0], data[:, 1], s=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_file, bbox_inches='tight')
    plt.cla()
    plt.close(fig)


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



class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


'''
helper functions from SDE
'''

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
'''
