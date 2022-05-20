import torch.nn as nn
from .base import _get_act


class FCN(nn.Module):
    def __init__(self, layers, activation='gelu'):
        super(FCN, self).__init__()
        self.activation = _get_act(activation)
        self.layer_config = layers
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size)
                                     for in_size, out_size in zip(layers, layers[1:])])

    def forward(self, x):
        length = len(self.layers)
        for i, linear in enumerate(self.layers):
            x = linear(x)
            if i != length - 1:
                x = self.activation(x)
        return x
