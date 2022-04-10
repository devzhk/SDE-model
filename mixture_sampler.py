import time
import random
import os

import numpy as np

import torch
import torch.distributions as D

from torchdiffeq import odeint_adjoint

from utils.helper import kde


class Gmm_score(object):
    def __init__(self, mu=1.0, num_modes=2, sigma2=0.01, beta_min=0.1, beta_max=20.):

        self.mus = np.linspace(-mu, mu, num_modes)
        print(f'mu: {mu}, num_modes: {num_modes}, sigma2: {sigma2}')

        assert len(self.mus) == num_modes
        self.num_modes = num_modes

        # self.mus = [[mu, mu], [-mu, mu], [-mu, -mu], [mu, -mu]]
        self.sigma2 = sigma2
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alphas_t = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)

    def sample(self, t, batch_size):
        assert isinstance(t, torch.Tensor) and t.ndim == 0

        mus = torch.tensor(self.mus).float().to(t.device)  # [num_modes, 2]
        alpha_t = self.alphas_t(t)
        mus_t = mus * torch.sqrt(alpha_t)  # [num_modes, 2]

        sigma2s_t = (1 - (1 - self.sigma2) * alpha_t) * torch.ones_like(mus).float()  # [num_modes, 2]
        mix_weights = torch.ones(self.num_modes,).to(t.device)
        # mix_weights = torch.tensor([1.0, 2], device=t.device)
        mix = D.Categorical(mix_weights)
        comp = D.Independent(D.Normal(loc=mus_t, scale=torch.sqrt(sigma2s_t)), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        samples = gmm.sample([batch_size])  # [bs, 2]

        '''
        # calculate score numerically
        samples_v2 = samples.detach().clone().requires_grad_(True)
        log_prob_gmm = gmm.log_prob(samples_v2)  # [bs]
        # score = torch.autograd.grad(log_prob_gmm, samples_v2, grad_outputs=torch.ones_like(log_prob_gmm))[0]
        score = torch.autograd.grad(log_prob_gmm.sum(), samples_v2)[0]
        print(f'score [numerical]: {score}')
        print(f'log_prop [numerical]: {log_prob_gmm}')
        '''

        return samples

    def _get_gaussian_at_t_i(self, t, idx):
        mu_i = torch.tensor(self.mus[idx]).float().to(t.device)  # [2]
        alpha_t = self.alphas_t(t)
        mu_t_i = mu_i * torch.sqrt(alpha_t)  # [2]
        Sigma2_t = (1 - (1 - self.sigma2) * alpha_t) * torch.eye(1).to(t.device)

        d_gs_t_i = D.Normal(mu_t_i, torch.sqrt(Sigma2_t))
        return d_gs_t_i

    def _get_gs_score_at_t_i(self, t, idx, x_t):
        mu_i = torch.tensor(self.mus[idx]).float().to(t.device)  # [2]
        alpha_t = self.alphas_t(t)
        mu_t_i = mu_i * torch.sqrt(alpha_t)  # [2]
        sigma2_t = 1 - (1 - self.sigma2) * alpha_t

        return - (x_t - mu_t_i.unsqueeze(0)) / sigma2_t

    def score(self, t, x_t):
        # x_t = self.sample(t, batch_size)  # [bs, 2]

        probs = []
        gs_scores = []
        for idx in range(self.num_modes):
            gaussian_t_i = self._get_gaussian_at_t_i(t, idx)
            prob_i = torch.exp(gaussian_t_i.log_prob(x_t))  # [bs]
            probs.append(prob_i)

            gs_score_i = self._get_gs_score_at_t_i(t, idx, x_t)  # [bs, 2]
            gs_scores.append(gs_score_i)

        # print(gs_scores)
        score = sum([p * s for (p, s) in zip(probs, gs_scores)]) / sum(probs)  # [bs, 2]

        '''
        # log prob
        log_prop = torch.log(sum(probs)) + torch.log(torch.tensor(0.25))
        print(f'log_prop: {log_prop}')
        '''

        # return score.detach().clone()
        return score


class VPODE(torch.nn.Module):
    def __init__(self, gmm_score):
        """Construct a Variance Preserving SDE.
        Args:
          model: diffusion model
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()

        self.gmm_score = gmm_score
        self.beta_0 = gmm_score.beta_min
        self.beta_1 = gmm_score.beta_max

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def ode_fn(self, t, x):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)

        score = self.gmm_score.score(t, x)

        ode_coef = drift - 0.5 * diffusion ** 2 * score
        return ode_coef

    def forward(self, t, states):
        x = states[0]

        with torch.set_grad_enabled(True):
            assert isinstance(t, torch.Tensor) and t.ndim == 0
            dx_dt = self.ode_fn(t, x)
            assert dx_dt.shape == x.shape

        return dx_dt,


class OdeDiffusion(torch.nn.Module):
    def __init__(self, args, img_shape=[1], gmm_score=None, device=None):
        super().__init__()
        self.args = args
        self.device = device
        self.step_size = args.step_size

        if gmm_score is None:
            print('using gmm_score with default settings!')
            gmm_score = Gmm_score()

        self.vpode = VPODE(gmm_score)

        self.atol, self.rtol = 1e-3, 1e-3
        self.method = 'euler'

        self.img_shape = img_shape

        print(f'method: {self.method}, atol: {self.atol}, rtol: {self.rtol}, step_size: {self.step_size}')

    def uncond_sample(self, batch_size, bs_id=-1, tag=None):
        shape = (batch_size, *self.img_shape)

        start_time = time.time()
        print("Start sampling in sample...")
        if tag is None:
            tag = 'rnd' + str(random.randint(0, 10000))
        out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

        if -1 < bs_id < 2:
            os.makedirs(out_dir, exist_ok=True)

        e = torch.FloatTensor(*shape).normal_(0, 1).to(self.device)
        x = e

        epsilon = 1e-5
        t0, t1 = 1., epsilon
        t_size = self.args.t_size
        ts = torch.linspace(t0, t1, t_size).to(self.device)

        x_ = x.view(batch_size, -1)  # (batch_size, state_size)
        states = (x_, )

        # ODE solver
        odeint = odeint_adjoint
        state_t = odeint(
            self.vpode,
            states,
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
            options=None if self.method != 'euler' else dict(step_size=self.step_size)  # only used for fixed-point method
        )  # 'euler', 'dopri5'

        # x0_ = state_t[0][-1]
        # x0 = x0_.view(x.shape)  # (batch_size, c, h, w)

        state_t = [x_i.view(x.shape).detach().clone() for x_i in state_t[0]]
        assert len(state_t) == t_size, len(state_t)

        if -1 < bs_id < 2:
            x0 = state_t[-1].cpu().numpy()
            kde(x0[:, 0], x0[:, 1], save_file=f'{out_dir}/samples0.png')
            xinit = state_t[0].cpu().numpy()
            kde(xinit[:, 0], xinit[:, 1], save_file=f'{out_dir}/init.png')

        minutes, seconds = divmod(time.time() - start_time, 60)
        print("Sampling time: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        return state_t