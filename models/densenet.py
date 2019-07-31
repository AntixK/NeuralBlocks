import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralBlocks.blocks.denseblock import DenseBlock
from NeuralBlocks.blocks.convnormrelupool import ConvNormReLUPool


cfgs = {
        '161': {'growth_rate' : 32,
                'block_config' : (6,12,24,16),
                'in_features'  : 64},
        '169': {'growth_rate' : 32,
                'block_config' : (6,12,32,32),
                'in_features'  : 64},
        '201': {'growth_rate' : 32,
                'block_config' : (6,12,48,32),
                'in_features'  : 64},
        }

class DenseNet(nn.Module):

    def __init__(self, in_channels, num_classes, in_features = 64, growth_rate = 32,
                 block_config=(6,12,24,16),bn_size=4, drop_rate=0, norm='BN'):
        super(DenseNet, self).__init__()

        self.features = ConvNormReLUPool(in_channels, in_features, conv_kernel_size=7,
                                         conv_stride=2, conv_padding=3, conv_bias=False,
                                         norm=norm, pool_kernel_size=3, pool_stride=2,
                                         pool_padding=1, pool_type='max')

        num_features = in_features

        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(num_layers,num_features,
                                     growth_rate=growth_rate,
                                     bn_size=bn_size,
                                     drop_rate=drop_rate, norm=norm)
            self.features.add_module('dense_block%d'%(i+1), dense_block)

            num_features += num_layers*growth_rate

            if i != len(block_config)-1:
                transition_block = ConvNormReLUPool(
                    num_features, num_features//2, conv_kernel_size=1, conv_stride=1,
                    conv_bias=False, norm=norm,pool_type='avg', pool_kernel_size=1,
                    pool_stride=1, conv_last=True)
                self.features.add_module('transition%d'%(i+1), transition_block)
                num_features = num_features // 2

        self.features.add_module('lastnorm', nn.BatchNorm2d(num_features))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (4,4)).view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    input = torch.randn(16,3,32,32).cuda()

    d = DenseNet(3, 10, norm='MSN').cuda()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        d(input)
    del d
    print(prof)




