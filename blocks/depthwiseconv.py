import torch.nn as nn

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
    def __init__(self, in_channels, out_channels, norm='BN', act = 'ReLU', **kwargs):
        super(DepthwiseSperableConv, self).__init__()

        if 'conv_args' in kwargs:
            kernel_size, stride,padding, bias = kwargs['conv_args']
            self.depth_conv = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding, groups=in_channels, bias=False)

        else:
            self.depth_conv = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1, groups=in_channels,bias=False)

        if norm == 'IN':
            self.norm_1 = nn.InstanceNorm2d(out_channels)
            self.norm_2 = nn.InstanceNorm2d(out_channels)

        else:
            if norm != 'BN':
                raise UserWarning('Undefined normalization '+norm+'. Using BatchNorm instead.')
            self.norm_1 = nn.BatchNorm2d(out_channels)
            self.norm_2 = nn.BatchNorm2d(out_channels)

        if act == 'LeakyReLU':
            self.act_1 = nn.LeakyReLU(inplace =True)
            self.act_2 = nn.LeakyReLU(inplace =True)
        else:
            self.act_1 = nn.ReLU(inplace=True)
            self.act_2 = nn.ReLU(inplace=True)

        self.point_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0, bias=False)

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.act_1(self.norm_1(x))

        x = self.point_conv(x)
        x = self.act_2(self.norm_2(x))

        return x

