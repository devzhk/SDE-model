import numpy as np
import torch


def ode_loss(u, ts, vpode, criterion, num_pad, L):
    '''
    Args:
        - u: (batchsize, T, d)
        - ts: (T, )
        - vpode: VPODE class
        - criterion: loss function
        - num_pad: number of padding points
        - L: period length
    '''
    T = u.shape[1]
    B = u.shape[0]
    d = u.shape[2]
    uhat = torch.fft.fft(u, dim=1)
    k_max = T // 2
    k_t = torch.cat((torch.arange(start=0, end=k_max, device=u.device),
                     torch.arange(start=-k_max, end=0, device=u.device)), 0)
    k_t = k_t.reshape(1, T, 1).to(u.device)
    ut_hat = 2j * np.pi / L * k_t * uhat
    ut = - torch.fft.irfft(ut_hat[:, :T//2+1, :], dim=1, n=T)

    t = ts.repeat(B).unsqueeze(-1)
    x = u[:, :-num_pad, :].reshape(-1, d)
    ode_coef = vpode.ode_fn(t, x)
    loss = criterion(ut[:, :-num_pad, :], ode_coef.reshape(B, T - num_pad, d))
    return loss


def fdm_ode_loss(u, ts, vpode, criterion):
    '''
    Args:
        - u : (B, T+2, d)
        - ts: (T,)
        - vpode: VPODE class
    '''
    T = ts.shape[0]
    B = u.shape[0]
    d = u.shape[2]
    dt = 2 / (T - 1)
    ut = (u[:, 2:] - u[:, 0:-2]) / dt

    t = ts.repeat(B).unsqueeze(-1)
    x = u[:, 1:-1, :].reshape(-1, d)
    ode_coef = vpode.ode_fn(t, x)
    loss = criterion(ut, ode_coef.reshape(B, T, d))
    return loss



