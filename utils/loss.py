import numpy as np
import torch


def ode_loss(u, xs, ts, vpode):
    '''
    Args:
        - u: (batchsize, T, d)
        - xs: (batchsize, T, d)
        - ts: (T, )
        - vpode: VPODE class
    '''
    T = u.shape[1]
    uhat = torch.fft.fft(u, dim=1)
    k_t = torch.fft.fftfreq(T).reshape(1, T, 1)
    ut_hat = 2j * np.pi * k_t * uhat
    ut = torch.fft.irfft(ut_hat[:, :T//2+1, :], dim=1, n=T)

    t = ts.repeat(xs.shape[0]).unsqueeze(-1)
    x = xs.reshape(-1, xs.shape[2])
    ode_coef = vpode.ode_fn(t, x)





