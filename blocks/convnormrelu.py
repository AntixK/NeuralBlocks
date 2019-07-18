import torch.nn as nn
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConvReLU
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConv2d
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConvReLU
from NeuralBlocks.blocks.spectralnorm import SpectralNormConv2d

class ConvNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = True, padding_mode='zeros', norm = 'BN',
                 groups_size=16, conv_last = False, act = 'relu'):
        super(ConvNormRelu, self).__init__()

        if norm not in [None,'BN', 'IN', 'GN', 'LN','WN', 'SN','MSN', 'MSNTReLU', 'WNTReLU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "[None,'BN', 'IN', 'GN', 'LN', 'WN', 'SN','MSN', 'MSNTReLU', 'WNTReLU']")

        def act_fn(act):
            if act == 'relu':
                return nn.ReLU(inplace=False)
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)
            else:
                raise ValueError('Undefined activation function.')

        layers = []
        if norm == 'MSN':
            conv2d = MeanSpectralNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, act_fn(act)]
        elif norm == 'MSNTReLU':
            conv2d = MeanSpectralNormConvReLU(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d]
        elif norm == 'SN':
            conv2d = SpectralNormConv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, act_fn(act)]
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
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.InstanceNorm2d(out_channels), act_fn(act)]
        elif norm == 'GN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.GroupNorm(groups_size, out_channels), act_fn(act)]
        elif norm == 'LN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.LayerNorm(groups_size, out_channels), act_fn(act)]
        elif norm == 'BN':
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups,
                 bias, padding_mode)
            layers += [conv2d, nn.BatchNorm2d(out_channels), act_fn(act)]
        else:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding, dilation, groups,
                               bias, padding_mode)
            layers += [conv2d, act_fn(act)]

        """
        conv_last is a flag to change the order of operations from
            Conv2D+ BN+ReLU to BN+ReLU+Con2D
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
        # print(self.layers)
        x = self.layers(input)
        return x
