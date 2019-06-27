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

        self.bias = nn.Parameter(torch.zeros(out_channels,1,1))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        x = self.conv(input)
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(1)[:, None, None]

        x = x - self.batch_mean + self.bias
        return x

class MeanWeightNormLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super(MeanWeightNormLinear, self).__init__()

        self.lin = weight_norm(nn.Linear(in_features, out_features, bias))

        self.bias = nn.Parameter(torch.zeros(out_features,1,1))

    def forward(self, input):

        x = self.lin(input)
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        if self.training:
            # Compute the mean and standard deviation
            self.batch_mean = x.mean(1)[:, None, None]

        x = x - self.batch_mean + self.bias
        return x
