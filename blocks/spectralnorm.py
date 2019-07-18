import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

class SpectralNormConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(SpectralNormConv2d, self).__init__()

        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)

        return x


class SpectralNormTransConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(SpectralNormTransConv2d, self).__init__()

        self.conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode, output_padding=output_padding))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)

        return x



###========================================================================================###
class SpectralNormLinear(nn.Module): # Basically BatchNorm 1D
    def __init__(self,in_features, out_features, bias=True):
        super(SpectralNormLinear, self).__init__()

        self.lin = spectral_norm(nn.Linear(in_features, out_features, bias))

    def forward(self, input):

        x = self.lin(input)
        return x

###========================================================================================###
