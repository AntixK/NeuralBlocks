import torch.nn as nn
from inplace_abn import ABN
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConv2d
from NeuralBlocks.blocks.spectralnorm import SpectralNormConv2d
from NeuralBlocks.blocks.weightnorm import WeightNormConv2d

class ConvNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros', norm = 'BN', groups_size=16, conv_last = False):
        super(ConvNorm, self).__init__()

        if norm not in [None,'BN', 'ABN','IN', 'GN', 'LN','WN', 'SN', 'MWN','MSN', 'MSNTReLU', 'MWNTReLU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "[None,'BN', 'ABN','IN', 'GN', 'LN', 'WN', 'SN','MWN', 'MSN', 'MSNTReLU', 'MWNTReLU']")
        layers = []
        if norm in ['MSN','MSNTReLU']:
            conv2d = MeanSpectralNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d ]
        elif norm == 'SN':
            conv2d = SpectralNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'WN':
            conv2d = WeightNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d ]
        elif norm in ['MWN', 'MWNTReLU']:
            conv2d = MeanWeightNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d ]
        elif norm == 'IN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.InstanceNorm2d(out_channels) ]
        elif norm == 'GN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.GroupNorm(groups_size, out_channels) ]
        elif norm == 'LN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.LayerNorm(out_channels) ]
        elif norm == 'BN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.BatchNorm2d(out_channels) ]
        elif norm == 'ABN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, ABN(out_channels) ]
        else:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding, dilation, groups,
                               bias, padding_mode)
            layers += [conv2d]

        """
            conv_last is a flag to change the order of operations from
                Conv2D+ BN to BN+Con2D
            This is frequently used in DenseNet & ResNet architectures.
            So to change the order, we simply rotate the array by 1 to the 
            left and change the num_features to the in_channels size
        """
        if conv_last:
            layers = layers[1:] + layers[:1]
            # Reinitialize the batchnorm layer or its variants
            if norm in ['BN', 'LN', 'IN', 'GN']:
                layers[0].__init__(in_channels)

        self.layers= nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x