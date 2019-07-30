import torch.nn as nn
from NeuralBlocks.blocks.convnorm import ConvNorm
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConvReLU
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConvReLU

class ConvNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros', norm = 'BN',
                 groups_size=16, conv_last = False, act = 'relu'):
        super(ConvNormRelu, self).__init__()

        if norm not in [None,'BN', 'ABN', 'IN', 'GN', 'LN','WN', 'SN','MWN','MSN', 'MSNTReLU', 'MWNTReLU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "[None,'BN', 'ABN','IN', 'GN', 'LN', 'WN', 'SN','MWN','MSN', 'MSNTReLU', 'MWNTReLU']")

        def act_fn(act):
            if act == 'relu':
                return nn.ReLU(inplace=False)
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            else:
                raise ValueError('Undefined activation function.')

        layers = []
        if norm == 'MSNTReLU':
            conv2d = MeanSpectralNormConvReLU(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'MWNTReLU':
            conv2d = MeanWeightNormConvReLU(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'ABN':
            conv2d = ConvNorm(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode, norm = norm, groups_size=groups_size, conv_last = conv_last)
            layers += [conv2d]
        else:
            conv2d = ConvNorm(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode, norm = norm, groups_size=groups_size, conv_last = conv_last)
            layers += [conv2d, act_fn(act)]
        self.layers= nn.Sequential(*layers)

    def forward(self, input):
        # print(self.layers)
        x = self.layers(input)
        return x
