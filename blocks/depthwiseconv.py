import torch.nn as nn
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class DepthwiseSperableConv(nn.Module):
    """
    A simple class for depthwise-separable convolution or
    simply "separable convolution. This block consists of

    1) depthwise convolution - convolution independently performed
                               for each channel
    2) Pointwise convolution - 1x1 convolution with the same number of
                               output channels from the previous
                               output.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias = False, padding_mode='zeros', norm='BN'):
        super(DepthwiseSperableConv, self).__init__()

        self.depth_conv = ConvNormRelu(in_channels, out_channels, kernel_size,
             stride, padding, dilation, groups,
             bias, padding_mode, norm)

        self.point_conv = ConvNormRelu(in_channels, out_channels, kernel_size=1,
             stride=1, bias=False, norm=norm)

    def forward(self, input):
        x = self.depth_conv(input)

        x = self.point_conv(x)

        return x

