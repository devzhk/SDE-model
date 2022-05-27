# %%
import numpy as np
import torch
import torch.nn.functional as F
# %%


class Gmm_score(object):
    def __init__(self, mu=2.0, num_modes=2, sigma2=0.01, beta_min=0.1, beta_max=20., dim=2):

        if dim == 1:
            self.mus = torch.linspace(-mu, mu, num_modes).unsqueeze(-1)
        elif dim == 2:
            xs = torch.linspace(-mu, mu, int(np.sqrt(num_modes)))
            ys = torch.linspace(-mu, mu, int(np.sqrt(num_modes)))
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            self.mus = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
        print(f'mu: {mu}, num_modes: {num_modes}, sigma2: {sigma2}')

        assert len(self.mus) == num_modes
        self.num_modes = num_modes

        self.sigma2 = sigma2
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alphas_t = lambda t: torch.exp(-0.5 *
                                            (beta_max - beta_min) * t**2 - beta_min * t)

    def score(self, t, x_t):
        mus = self.mus.to(x_t.device)
        alpha_t = self.alphas_t(t)
        mus_t = mus * torch.sqrt(alpha_t)  # [nm, 2]
        sigma2_t = 1 - (1 - self.sigma2) * alpha_t

        def pairwise_L2_square(a, b):
            b = b.T
            a2 = torch.sum(torch.square(a), dim=1, keepdim=True)
            b2 = torch.sum(torch.square(b), dim=0, keepdim=True)
            ab = torch.mm(a, b)
            return a2 + b2 - 2 * ab

        logits_t = - pairwise_L2_square(x_t, mus_t) / (2 * sigma2_t)

        def own_softmax(x):
            maxes = torch.max(x, 1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)
            x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
            return x_exp / x_exp_sum
        gmm_coef_t = own_softmax(logits_t)  # [bs, mn]
        score = -1 / sigma2_t * (x_t - torch.mm(gmm_coef_t, mus_t))
        return score

    def batch_score(self, ts, xs):
        # ts: (B, 1)
        # xs: (B, d)

        d = self.mus.shape[1]
        M = self.mus.shape[0]
        B = xs.shape[0]
        mus = self.mus.to(xs.device).reshape(1, M, d)
        alpha_t = self.alphas_t(ts).reshape(B, 1, 1)
        mus_t = mus * torch.sqrt(alpha_t)   # (B, M, d)
        sigma2_t = 1 - (1 - self.sigma2) * alpha_t  # (B, 1, 1)

        xs = xs.reshape(B, 1, d)
        diff = xs - mus_t   # (B, M, d)
        logits_t = - torch.sum(diff * diff, dim=-1) / \
            (2 * sigma2_t.squeeze(-1))  # (B, M)
        gmm_coef_t = F.softmax(logits_t, dim=1)  # (B, M)

        sum_mu = torch.einsum('bm,bmd->bd', gmm_coef_t, mus_t)  # (B, d)
        xs = xs.reshape(B, d)
        score = (sum_mu - xs) / sigma2_t.reshape(B, 1)
        return score


# %%
gmm = Gmm_score(mu=2, num_modes=4)
# %%
x = torch.tensor([[0.0, 0.0],
                  [1.0, 1.0]])

t = torch.tensor(0.6)
# %%
res = gmm.score(t, x)
print(res)
# %%
xs = x
ts = torch.tensor([[0.5], [0.6]])
res = gmm.batch_score(ts, xs)
print(res)
# %%
