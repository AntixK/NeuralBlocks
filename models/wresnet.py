import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralBlocks.blocks.resblock import ResidualBlock

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes,  widen_factor,dropout_rate = 0.2, norm='BN'):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.norm = norm
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'

        n = int((depth-4)/6)
        k = widen_factor
        num_stages = [16,16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, num_stages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wideLayer(num_stages[1], n,dropout_rate,stride=1)
        self.layer2 = self._wideLayer(num_stages[2], n,dropout_rate,stride=2)
        self.layer3 = self._wideLayer(num_stages[3], n,dropout_rate,stride=2)

        if self.norm == 'BN':
            self.bn = nn.BatchNorm2d(num_stages[3], momentum=0.9)

        self.fc = nn.Linear(num_stages[3], num_classes)

        #print(self.layer1)

    def _wideLayer(self, out_planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers= []

        for stride in strides:
            layers.append(ResidualBlock(self.in_planes,
                                        out_planes,
                                        dropout_rate=dropout_rate,
                                        stride=stride, padding=1, norm=self.norm, conv_last=True))
            self.in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.norm == 'BN':
            x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x,8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net=WideResNet(depth=28, num_classes=10, dropout_rate=0.3, widen_factor=10, norm=None)
    y = net(torch.randn(1,3,32,32))

    print(y.size())
