import torch.nn as nn
from NeuralBlocks.blocks.convnorm import ConvNorm

class ConvNormPool(nn.Module):
    """
    A simply block consisting of a convolution layer,
    a normalization layer and a pooling
    layer. For example, this is the first block in the
    ResNet architecture.
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                       conv_stride=1, conv_padding=1, conv_bias=False, norm='BN',
                       pool_type='max',pool_kernel_size=2, pool_stride=1,pool_padding=0):
        super(ConvNormPool, self).__init__()

        if pool_type not in ['max', 'avg']:
            raise ValueError("pool_type must be either 'max' or 'avg'.")

        self.conv_norm = ConvNorm(in_channels, out_channels, norm=norm,
                 kernel_size=conv_kernel_size, stride= conv_stride,
                 padding=conv_padding, bias= conv_bias)

        if pool_type is 'max':
            self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size,
                                    stride=pool_stride, padding=pool_padding)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=pool_kernel_size,
                                        stride=pool_stride, padding=pool_padding)


    def forward(self, input):
        x = self.conv_norm(input)
        x = self.maxpool(x)
        return x
