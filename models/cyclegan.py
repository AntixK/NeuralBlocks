import torch.nn as nn
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu
from NeuralBlocks.blocks.resblock import ResidualBlock
from NeuralBlocks.blocks.transconvnormrelu import TransConvNormRelu


class ResNetGenerator(nn.Module):

    def __init__(self, in_channels, num_res_blocks, out_channels, norm='BN'):

        super(ResNetGenerator, self).__init__()

        layers = []

        out_features = 64
        layers.append(nn.ReflectionPad2d(in_channels))
        layers.append(ConvNormRelu(in_channels, out_features, kernel_size=7, norm = norm))


        #Down sampling
        in_features = out_features
        out_features = in_features*2
        for _ in range(2):
            layers.append(ConvNormRelu(in_features,out_features, kernel_size=3, stride=2,
                                       padding=1, norm=norm))
            in_features = out_features
            out_features = in_features*2

        # Residual Blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(in_channels=in_features,
                                        out_channels=in_features,norm=norm,
                                        reflection_pad = 1))
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):

            layers.append(TransConvNormRelu(in_features,out_features,kernel_size=3, stride=2,
                            padding=1, output_padding=1,norm=norm))
            in_features = out_features
            out_features = in_features // 2
            # print(in_features, out_features)

        #Output Layer
        layers += [ nn.ReflectionPad2d(in_channels),
                    nn.Conv2d(64, out_channels, 7),
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
                        ConvNormRelu(64,128, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        ConvNormRelu(128,256, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        ConvNormRelu(256,512, kernel_size=4, stride=2,
                                     padding=1, norm=norm,act = 'leakyrelu'),
                        nn.ZeroPad2d((1, 0, 1, 0)),
                        nn.Conv2d(512, 1, 4, padding=1)

                        )

    def forward(self, input):
        return self.layers(input)

if __name__ == "__main__":
    # u = ResNetGenerator(in_channels=3, num_res_blocks=9, out_channels=3, norm='BN')
    u = Discriminator(in_channels=3)
    import torch
    inp = torch.randn(1,3,256,256) #M x C x H x W
    u.train()
    result = u(inp)

