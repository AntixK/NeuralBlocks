import torch.nn as nn
from NeuralBlocks.blocks.resblock import InvertedResidualBlock
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class MobileNetV1(nn.Module):

    def __init__(self , num_class = 10, width_mult=1., norm='BN'):
        super(MobileNetV1, self).__init__()

#====================================================================#
# (Expansion, out_channels, num_blocks, stride)
# CFG = [(1,16,1,1),
#        (6,24,2,1),
#        (6,32,3,2),
#        (6,64,4,2),
#        (6,96,3,1),
#        (6,320,1,1)]

CFG = [(1,16,1,1),
       (6,24,2,1)]

class MobileNetV2(nn.Module):
    """
    MobileNet V2 is very similar to ResNets
    but the residual layers have an expansion layer
    and a depthwise separable layer.
    """

    def __init__(self , in_channels, num_class ,norm='BN', config=None):
        super(MobileNetV2, self).__init__()

        if config is None:
            self.cfg = CFG
        else:
            self.cfg = config
        self.norm = norm
        self.conv1 = ConvNormRelu(in_channels, 32,kernel_size=3,
                              stride=1, padding=1, bias=False,norm=norm)

        self.invreslayers = self._invResLayer(in_channels=32)
        self.conv2 = ConvNormRelu(self.cfg[-1][1], self.cfg[-1][1]*2,kernel_size=1, stride=1,bias=False)
        self.pool = nn.AdaptiveAvgPool2d((5,5))
        self.linear = nn.Linear(self.cfg[-1][1]*2*5*5, num_class)

    def _invResLayer(self, in_channels):
        layers = []

        for expansion, out_channels, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)

            for stride in strides:
                layers.append(InvertedResidualBlock(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    expansion = expansion,
                                                    stride= stride, norm = self.norm))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.invreslayers(x)


        x = self.conv2(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)
        return x


if __name__ == '__main__':
    import torch
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # from torchsummary import summary
    net = MobileNetV2(in_channels=1, num_class=10).cuda()
    x = torch.randn(2,1,32,32).cuda()
    y = net(x)
    print(y.size())
    # print(summary(net, (1,28,28),batch_size=2, device="cpu"))



