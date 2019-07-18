import torch.nn as nn
from NeuralBlocks.blocks.convnorm import ConvNorm
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm='BN', kernel_size=3,
                 stride=1,padding=0, reflection_pad = 0, dropout_rate =0, conv_last = False):
        super(ResidualBlock, self).__init__()
        modules= []
        modules.append(ConvNormRelu(in_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=1, padding=padding, conv_last=conv_last))
        modules.append(ConvNorm(out_channels, out_channels, norm=norm, kernel_size=kernel_size,
                             stride=stride, padding=padding, conv_last= conv_last))

        if reflection_pad > 0:
            modules.insert(0, nn.ReflectionPad2d(reflection_pad))
            modules.insert(2, nn.ReflectionPad2d(reflection_pad))

        # Dropout for the residual block is used for WideResNets
        if dropout_rate > 0:
            modules.insert(2, nn.Dropout(p=dropout_rate))

        self.layer = nn.Sequential(*modules)

        """
            For the residual part, the input is usually added to the
            output of the block. But, if the stride is > 1 or the out_channels !=
            in_channels, then there will be a size mis-match.
            To resolve this, either pad the input to the size of the layer output
            or the more commonly used one - 1x1 convolution so that the resultant
            sizes are the same.
        """

        if stride !=1 or out_channels != in_channels:
            self.residue = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                   stride=stride, bias=True))
        else:
            self.residue = nn.Sequential()


    def forward(self, input):
        residual = self.residue(input)
        x =  self.layer(input)
        return residual + x