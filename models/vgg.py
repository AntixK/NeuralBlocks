import torch.nn as nn
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConvReLU
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConv2d
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormConvReLU
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
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

        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
                            nn.Linear(512*7*7,4096),
                            nn.ReLU(True),
                            nn.Dropout(),
                            nn.Linear(4096,4096),
                            nn.ReLU(True),
                            nn.Dropout(),
                            nn.Linear(4096, num_class))
        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, MeanSpectralNormConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.conv.bias is not None:
                    nn.init.constant_(m.conv.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

