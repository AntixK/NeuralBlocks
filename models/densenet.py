import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, num_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.BatchNorm2d(num_features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(num_features, bn_size*growth_rate,
                                 kernel_size=1, stride=1, bias=False)
                        )

        self.layer2 = nn.Sequential(
                        nn.BatchNorm2d(bn_size*growth_rate),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        x = torch.cat(input,1)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.drop_rate > 0:
            x = F.dropout(x, p =self.drop_rate, training=self.training)

        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_features + i*growth_rate, growth_rate=growth_rate,
                               bn_size=bn_size, drop_rate=drop_rate)
            self.add_module('denselayer{}'.format(i+1), layer)

    def forward(self, input):
        fea


class DenseNet:
    pass