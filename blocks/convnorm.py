import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

class ConvNorm(nn.Module):
    """
    A simply block consisting of a convolution layer,
    a normalization layer, ReLU activation and a pooling
    layer. For example, this is the first block in the
    ResNet architecture.
    """

    def __init__(self, in_channels, out_channels,norm='BN', **kwargs):
        super(ConvNorm, self).__init__()

        if 'conv_args' in kwargs:
            kernel_size, stride,padding, bias = kwargs['conv_args']
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding, bias=bias)

        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=3, bias=False)

        self.norm_type = norm

        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(16, out_channels)
        else:
            if norm != 'BN':
                raise UserWarning('Undefined normalization '+norm+'. Using BatchNorm instead.')
            self.norm = nn.BatchNorm2d(out_channels)


    def forward(self, input):
        x = self.norm(self.conv(input))
        return x

if __name__ == "__main__":
    import torch
    u = ConvNorm(3, 10,conv_args=(3,1,0, True))

    inp = torch.randn(32,3,128,128) #M x C x H x W
    u.train()
    result = u(inp)
    print(result.size())