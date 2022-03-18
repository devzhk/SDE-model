import torch.nn as nn
import torch.nn.functional as F
from .base import SpectralConv1d, SpectralConv2d
'''
TimeConv
'''

class TimeConv(nn.Module):
    def __init__(self,
                 modes, width=32,
                 layers=None,
                 fc_dim=128,
                 in_dim=3, out_dim=1,
                 activation='relu'):
        super(TimeConv, self).__init__()
        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4
        # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, num_modes) for in_size, out_size, num_modes in zip(layers, layers[1:], self.modes1)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        length = len(self.ws)

        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)

        x = x.permute(0, 2, 1)
        return x
