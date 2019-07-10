import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

class MeanSpectralNormConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormConv2d, self).__init__()

        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        x = self.conv(input)
        size = x.size()
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(1)[:, None]

        x = x - self.batch_mean + self.bias
        return x.view(size)

class MeanSpectralNormTransConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormTransConv2d, self).__init__()

        self.conv = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode, output_padding=output_padding))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        x = self.conv(input)
        size = x.size()
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(1)[:, None]

        x = x - self.batch_mean + self.bias
        return x.view(size)
###========================================================================================###
class MeanSpectralNormConvReLU(nn.Module):
    """
    Implements Conv2d layer with MSN followed by a modified ReLU layer called as
    'Translated ReLU' from the paper 'https://arxiv.org/abs/1704.03971'
    """
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormConvReLU, self).__init__()

        self.msn = MeanSpectralNormConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.msn(input)
        x = self.relu(x) - self.bias
        return x

class MeanSpectralNormTransConvReLU(nn.Module):
    """
    Implements Conv2d layer with MSN followed by a modified ReLU layer called as
    'Translated ReLU' from the paper 'https://arxiv.org/abs/1704.03971'
    """
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanSpectralNormTransConvReLU, self).__init__()

        self.msn = MeanSpectralNormTransConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.msn(input)
        x = self.relu(x) - self.bias
        return x
###========================================================================================###
class MeanSpectralNormLinear(nn.Module): # Basically BatchNorm 1D
    def __init__(self,in_features, out_features, bias=True):
        super(MeanSpectralNormLinear, self).__init__()

        self.lin = spectral_norm(nn.Linear(in_features, out_features, bias))

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):

        x = self.lin(input)

        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(0)

        x = x - self.batch_mean + self.bias
        return x

###========================================================================================###
class MeanSpectralNormLinReLU(nn.Module):
    """
    Implements Linear layer with MSN followed by a modified ReLU layer called as
    'Translated ReLU' from the paper 'https://arxiv.org/abs/1704.03971'
    """
    def __init__(self,in_features, out_features, bias=True):
        super(MeanSpectralNormLinReLU, self).__init__()

        self.msn = MeanSpectralNormLinear(in_features, out_features, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.msn(input)
        x = self.relu(x) - self.bias
        return x