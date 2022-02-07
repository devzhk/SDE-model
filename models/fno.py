import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

################################################################
# 2d fourier layer
################################################################


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-(k2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:,:,1:,1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:,:,0:1,1:].flip(-1).conj(), coeff2[:,:,:,1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y


class FNN2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 activation='tanh'):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation == F.relu
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        x = x.permute(0, 2, 3, 1)   # batchsize x 1 x 28 x 28 -> batchsize x 28 x 28 x 1
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        x = self.fc0(x)             # batchsize x 28 x 28 x 32
        x = x.permute(0, 3, 1, 2)   # batchsize x 32 x 28 x 28

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i + 1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 3, 1)   # batchsize x 28 x 28 x 32
        x = self.fc1(x)             # batchsize x 28 x 28 x 1
        x = self.activation(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x