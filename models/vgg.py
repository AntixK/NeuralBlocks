import torch.nn as nn
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConvReLU
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConv2d
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConvReLU
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 128, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, model_config, in_channels, num_class, norm = 'BN', init_weights=False):
        super(VGG, self).__init__()
        layers= []

        if not isinstance(model_config, list):
            if model_config in [11,13,16,19]:
                model_config = cfgs[model_config]
            else:
                raise ValueError("Invalid model_config. Must be a list of configs or a value"
                                 "in [11,13,16,19]")


        for v in model_config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers+=[ConvNormRelu(in_channels, v, kernel_size=3, padding=1, norm=norm)]
                in_channels = v
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((3,3))

        self.classifier = nn.Sequential(nn.Linear(128*3*3, num_class))
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    net =  VGG(11,1, num_class=10, norm='BN')
    import torch
    input= torch.randn(32,1,28,28)
    print(net(input).size())
