import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam

from models.ddpm import Generator


def marginal_prob(x, t, beta_min=0.1, beta_max=20):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.

    Returns:
      The standard deviation.
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    mean = torch.exp(log_mean_coeff[:, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


def loss_fn(model, x, eps=1e-3):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    mean, std = marginal_prob(x, random_t)

    perturbed_x = mean + z * std[:, None]
    score = model(perturbed_x, random_t)

    losses = torch.square(score - z)
    # losses = torch.square(score*std[:,None] + z)
    losses = torch.sum(losses, dim=1)  # * weight
    loss = torch.mean(losses)
    return loss


def inf_train_gen(DATASET, BATCH_SIZE):
    if DATASET == '25gaussians':

        dataset = []
        for i in range(1000000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in range(len(dataset) // BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .08
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
score_model = Generator()
score_model = score_model.to(device)

n_iterations = 200000
## size of a mini-batch
batch_size = 512
## learning rate
lr = 1e-3

sampler = inf_train_gen('25gaussians', batch_size)

optimizer = Adam(score_model.parameters(), lr=lr)

data_list = []
for iteration in tqdm(range(n_iterations)):

    avg_loss = 0.
    num_items = 0
    real_data = sampler.__next__()
    x = torch.Tensor(real_data).to(device)
    # x = sampler.spl(batch_size).to(device)

    data_list.append(x)

    loss = loss_fn(score_model, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
    if iteration % 1000 == 0:
        print('iteration{}, Average Loss: {:5f}'.format(iteration, avg_loss / num_items))

torch.save(score_model.state_dict(), 'exp/ref_ddpm_25gm.pt')

