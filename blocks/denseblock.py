import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class DenseBlock(nn.Module):
    """
    DenseBlock consists of Denselayers.
    A denselayer consists of two layers of BN+ReLU+Conv.
    Dropout after conv is optional.
    """
    def __init__(self, num_layers, in_features, growth_rate, bn_size, drop_rate, norm='BN'):
        super(DenseBlock, self).__init__()


        for i in range(num_layers):
            dense_layer = nn.Sequential(
                            ConvNormRelu(in_features + i * growth_rate, bn_size*growth_rate,
                                   kernel_size=1, stride=1, bias=False,norm=norm, conv_last=True),
                            ConvNormRelu(bn_size*growth_rate, growth_rate,
                                   kernel_size=3, stride=1, padding=1, bias=False, norm=norm, conv_last=True))

            self.add_module('dense_layer%d'%(i+1), dense_layer)

        self.drop_rate = drop_rate

    def forward(self, inputs):
        features = [inputs]
        for name, layer in self.named_children():
            x = torch.cat(features, 1)
            x = layer(x)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            features.append(x)
        return x


