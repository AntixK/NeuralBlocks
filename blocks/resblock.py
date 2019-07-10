import torch.nn as nn
import torch.nn.functional as F
from NeuralBlocks.blocks.convnorm import ConvNorm
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm='BN', kernel_size=3,
                 stride=1,padding=0, reflection_pad = 0):
        super(ResidualBlock, self).__init__()

        if reflection_pad > 0:
            self.layer = nn.Sequential(
                nn.ReflectionPad2d(reflection_pad),
                ConvNormRelu(in_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=stride, padding=padding),
                nn.ReflectionPad2d(reflection_pad),
                ConvNorm(out_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=stride, padding=padding))
        else:
            self.layer = nn.Sequential(
                ConvNormRelu(in_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=stride, padding=padding),
                ConvNorm(out_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=stride, padding=padding))

    def forward(self, input):
        residual = input
        return F.relu(residual + self.layer(input))