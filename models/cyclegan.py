import torch.nn as nn
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu
from NeuralBlocks.blocks.resblock import ResidualBlock


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, num_res_blocks, out_channels = 64, norm='BN'):

        super(ResNetGenerator, self).__init__()

        layers = []

        layers.append(nn.ReflectionPad2d(in_channels))
        layers.append(ConvNormRelu(in_channels, out_channels, kernel_size=7, norm = norm))

        in_channels = out_channels

        #Down sampling
        for _ in range(2):
            out_channels *= 2
            layers.append(ConvNormRelu(in_channels,out_channels, kernel_size=3, stride=2,
                                       padding=1, norm=norm))
            in_channels = out_channels

        # Residual Blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(in_channels=out_channels,
                                        out_channels=out_channels,norm=norm,
                                        reflection_pad = 1))
        # Upsampling
        for _ in range(2):
            out_channels = out_channels // 2
            layers += [
                        nn.Upsample(scale_factor=2),
                        ConvNormRelu(in_channels,out_channels,kernel_size=3, stride=1,
                                     padding=1, norm=norm)
                       ]
            in_channels = out_channels

        #Output Layer
        layers += [ nn.ReflectionPad2d(in_channels),
                    nn.Conv2d(out_channels, in_channels, 7),
                    nn.Tanh()]

        self.layers= nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class Discriminator(nn.Module):

    def __init__(self, in_channels, norm='BN'):

        super(Discriminator, self).__init__()

        self.layers= nn.Sequential(
                        ConvNormRelu(in_channels, 64, kernel_size=4, stride=2,
                                     padding=1, norm=None,act = 'leakyrelu'),
                        ConvNormRelu(in_channels, 64, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        ConvNormRelu(in_channels, 64, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        ConvNormRelu(in_channels, 64, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        nn.ZeroPad2d(1, 0, 1, 0),
                        nn.Conv2d(512, 1, 4, padding=1)

                        )

    def forward(self, input):
        return self.layers(input)



