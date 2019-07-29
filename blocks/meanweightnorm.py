import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

"""
    Implements mean-only weight norm from the "
    Weight Normalization paper by Salimans and Kingma
    arXiv - https://arxiv.org/abs/1602.07868
"""

class MeanWeightNormConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanWeightNormConv2d, self).__init__()

        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        x = self.conv(input)
         # Recenter the pre-activations using running mean
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            y = y - mu.view(-1, 1)

        y = y + self.bias
        return y.view(return_shape).transpose(0, 1)

class MeanWeightNormTransConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanWeightNormTransConv2d, self).__init__()

        self.conv = weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups,
                              bias=bias, padding_mode=padding_mode, output_padding=output_padding))

        self.bias = nn.Parameter(torch.zeros(out_channels,1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.momentum = 0.1

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        x = self.conv(input)
         # Recenter the pre-activations using running mean
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            y = y - mu.view(-1, 1)

        y = y + self.bias
        return y.view(return_shape).transpose(0, 1)

class MeanWeightNormConvReLU(nn.Module):
    """
    Implements Conv2d layer with Weight Normalization followed by a
    modified ReLU layer called as 'Translated ReLU'
    from the paper 'https://arxiv.org/abs/1704.03971'
    """
    def __init__(self,in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros'):
        super(MeanWeightNormConvReLU, self).__init__()

        self.wn = MeanWeightNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias , padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.wn(input)
        x = self.relu(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        y = y - self.wn.bias
        return y.view(return_shape).transpose(0, 1)

class MeanWeightNormLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super(MeanWeightNormLinear, self).__init__()

        self.lin = weight_norm(nn.Linear(in_features, out_features, bias))

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):

        x = self.lin(input)

        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(0)

        x = x - self.batch_mean + self.bias
        return x

class MeanWeightNormLinReLU(nn.Module):
    """
    Implements linear layer with Weight Normalization followed by a
    modified ReLU layer called as 'Translated ReLU'
    from the paper 'https://arxiv.org/abs/1704.03971'
    """
    def __init__(self,in_features, out_features, bias=True):
        super(MeanWeightNormLinReLU, self).__init__()

        self.wn = MeanWeightNormLinear(in_features, out_features, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.wn(input)
        x = self.relu(x) - self.wn.bias
        return x
