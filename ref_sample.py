import matplotlib.pyplot as plt
import numpy as np

import torch

import torch.nn as nn
from utils.helper import kde, scatter

torch.manual_seed(2022)

DIM = 512  # Model dimensionality

# ==================Definition Start======================
# %%
beta_min = 0.1
beta_max = 20


def var_func(t):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


# %%
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

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
        cat_input = torch.cat([x_t, t[:, None]], dim=1)
        output = self.main(cat_input)
        return output


# Don't need these for sampling, unless you want to train a score based model
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def plot_sample(y, fname, title):
    plt.style.use('seaborn-darkgrid')
    plt.figure()
    plt.title(title, fontsize=20)
    plt.scatter(y[:, 0], y[:, 1], alpha=0.5, s=0.5, color='blue', marker='o', label='y')
    plt.savefig(fname)
    plt.close()


# %% sampling function
import math


def make_beta_schedule_correct(n_timestep):
    t = np.arange(1, n_timestep + 1, dtype=np.float32)
    t = t / n_timestep
    t = torch.from_numpy(t)
    # alpha_bars = self._alpha_bars_fun(t)
    var = var_func(t)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = 1 - alpha_bars[0]
    betas = torch.cat((first[None], betas))
    return betas


def extract(input, t, shape):  # utility function for select indices
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def sample_from_model(generator, batch_size, device, n_time=100, scheduler='cosine'):
    betas = make_beta_schedule_correct(n_time)
    betas = betas.to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    alphas_cumprod_prev = torch.cat(
        (torch.tensor([1.], dtype=torch.float32, device=device), alphas_cumprod[:-1]), 0
    )
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.rsqrt(alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / alphas_cumprod - 1)

    posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
    posterior_mean_coef2 = ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))

    posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

    def predict_start_from_noise(x_t, t, noise):  # given x_t and eps, predict x_0
        return (extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior(x_0, x_t, t):  # mean and variance for q(x_t-1|x_t,x_0)
        mean = (
                extract(posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(posterior_variance, t, x_t.shape)
        log_var_clipped = extract(posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_mean_variance(generator, x_t, t):
        # t is integer indices, here convert it to time step in [0,1] and input to the score model
        t_time = torch.ones(batch_size, dtype=torch.float32, device=device) * betas[t[0]]
        x_recon = predict_start_from_noise(x_t, t, noise=generator(x_t, t_time))

        mean, var, log_var = q_posterior(x_recon, x_t, t)

        return mean, var, log_var

    def p_sample(generator, x_t, t):
        mean, _, log_var = p_mean_variance(generator, x_t, t)

        noise = torch.randn_like(x_t)

        # no noise at the last step
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None] * torch.exp(0.5 * log_var) * noise

    def p_sample_loop(generator, batch_size):
        img = torch.randn(batch_size, 2).to(device)

        for i in reversed(range(n_time)):
            img = p_sample(
                generator,
                img,
                torch.full((batch_size,), i, dtype=torch.int64).to(device)
            )

        return img

    sample = p_sample_loop(generator, batch_size)

    return sample


# %%
# ==================Definition End======================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = {
    'temb_dim': 1,
    'x_dim': 2
}
# netG = DDPM(config).to(device)
# ckpt = torch.load('exp/ddpm-01/ckpts/score_final.pt', map_location=device)
netG = Generator().to(device)
ckpt = torch.load('exp/ref_ddpm_25gm.pt', map_location=device)
netG.load_state_dict(ckpt)
netG = netG

batchsize = 500

sample_list = []
for i in range(9):
    sample_generated = sample_from_model(netG, batchsize, device, n_time=500)
    sample_list.append(sample_generated)

data = torch.cat(sample_list, dim=0)
scatter(data.detach().cpu().numpy(), 'figs/ref_ddpm_sample.png')
# plot_sample(data.detach().cpu().numpy(), 'figs/ref_ddpm_sample.png', 'sample score (discrete)')
# kde(data, 'figs/ref_ddpm_sampler.png', dim=2)
