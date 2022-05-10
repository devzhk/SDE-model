import time
import math
import torch

from torchdiffeq import odeint_adjoint


class VPODE(torch.nn.Module):
    def __init__(self, model, beta_min, beta_max, scale_sigma=False):
        """Construct a Variance Preserving SDE.
        Args:
          model: diffusion model
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        super().__init__()
        self.model = model
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alphas_t = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t ** 2 - beta_min * t)
        self.scale_sigma = scale_sigma

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def ode_fn(self, t, x):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)
        ts = t.repeat(x.shape[0])
        score = self.model(x, ts)
        if self.scale_sigma:
            score = - score / torch.sqrt(1 - self.alphas_t(t))

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
    def __init__(self, config, vpode, img_shape=[1], device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.step_size = config['step_size']
        self.vpode = vpode
        self.atol, self.rtol = 1e-3, 1e-3
        self.method = 'euler'

        self.img_shape = img_shape

        print(f'method: {self.method}, atol: {self.atol}, rtol: {self.rtol}, step_size: {self.step_size}')

    def uncond_sample(self, batch_size, bs_id=-1, tag=None):
        shape = (batch_size, *self.img_shape)

        start_time = time.time()
        print("Start sampling in sample...")

        e = torch.FloatTensor(*shape).normal_(0, 1).to(self.device)
        x = e

        epsilon = 1e-5
        t0, t1 = 1., epsilon
        t_size = self.config['t_size']
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

        minutes, seconds = divmod(time.time() - start_time, 60)
        print("Sampling time: {:0>2}:{:05.2f}".format(int(minutes), seconds))
        state_t = torch.stack(state_t, dim=1)
        return state_t