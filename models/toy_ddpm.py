import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(config):
    """Get activation functions from the config file."""

    if config['act'] == 'elu':
        return nn.ELU()
    elif config['act'] == 'relu':
        return nn.ReLU()
    elif config['act'] == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config['act'] == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        # self.actitvation = get_act(config)
        self.dim_temb = config['temb_dim']
        self.x_dim = x_dim = config['x_dim']
        DIM = 512
        self.net = nn.Sequential(
            nn.Linear(self.dim_temb + x_dim, DIM),
            nn.Softplus(),
            nn.Linear(DIM, DIM),
            nn.Softplus(),
            nn.Linear(DIM, DIM),
            nn.Softplus(),
            nn.Linear(DIM, x_dim)
        )

    def forward(self, x, c):
        # x: data, c: conditions
        # temb = get_timestep_embedding(c, self.dim_temb)
        feature = torch.cat([x, c[:, None]], dim=-1)
        out = self.net(feature)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        DIM = 512

        main = nn.Sequential(
            nn.Linear(3, DIM),
            nn.Softplus(),
            nn.Linear(DIM, DIM),
            nn.Softplus(),
            nn.Linear(DIM, DIM),
            nn.Softplus(),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, x_t, t):
        cat_input = torch.cat([x_t, t[:, None]], dim=-1)
        output = self.main(cat_input)
        return output
