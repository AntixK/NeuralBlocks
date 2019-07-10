import torch.nn as nn
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormTransConv2d
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormTransConvReLU

from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConv2d
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConvReLU

class TransConvNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias = True, padding_mode='zeros',
                 norm = 'BN', groups_size=16, conv_last = False,
                 act = 'relu'):
        super(TransConvNormRelu, self).__init__()

        if norm not in [None,'BN', 'IN', 'GN', 'LN','WN', 'MSN', 'MSNTReLU', 'WNTRelU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "[None,'BN', 'IN', 'GN', 'LN', 'WN', 'MSN', 'MSNTReLU', 'WNTRelU']")

        def act_fn(act):
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            else:
                raise ValueError('Undefined activation function.')

        layers = []
        if norm == 'MSN':
            conv2d = MeanSpectralNormTransConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, act_fn(act)]
        elif norm == 'MSNTReLU':
            conv2d = MeanSpectralNormTransConvReLU(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'WN':
            conv2d = MeanWeightNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, act_fn(act)]
        elif norm == 'WNTReLU':
            conv2d = MeanWeightNormConvReLU(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'IN':
            conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                 groups=groups, bias=bias, padding_mode=padding_mode)
            layers += [conv2d, nn.InstanceNorm2d(out_channels), act_fn(act)]
        elif norm == 'GN':
            conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                 groups=groups, bias=bias, padding_mode=padding_mode)
            layers += [conv2d, nn.GroupNorm(groups_size, out_channels), act_fn(act)]
        elif norm == 'LN':
            conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                 groups=groups, bias=bias, padding_mode=padding_mode)
            layers += [conv2d, nn.LayerNorm(groups_size, out_channels), act_fn(act)]
        elif norm == 'BN':
            conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                 groups=groups, bias=bias, padding_mode=padding_mode)
            layers += [conv2d, nn.BatchNorm2d(out_channels), act_fn(act)]
        else:
            conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                 stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                 groups=groups, bias=bias, padding_mode=padding_mode)
            layers += [conv2d, act_fn(act)]

        """
        conv_last is a flag to change the order of operations from
            Conv2D+ BN+ReLU to BN+ReLU+Con2D
        This is frequently used in DenseNet architectures.
        So to change the order, we simply rotate the array by 1 to the 
        left and change the num_features to the in_channels size
        """
        if conv_last:
            layers = layers[1:] + layers[:1]
            layers[0].num_features = in_channels

        self.layers= nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x