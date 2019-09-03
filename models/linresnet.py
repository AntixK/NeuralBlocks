import torch
import torch.nn as nn
import torch.nn.functional as F

from NeuralBlocks.blocks.resblock import LinearResidualBlock

class LinearResNet(nn.Module):
    def __init__(self, in_channels, num_classes, widen_factor, dropout_rate = 0, norm='BN'):
        super(LinearResNet, self).__init__()
        self.in_planes = 16*widen_factor
        self.norm = norm

        num_stages = [16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.lin1 = nn.Linear(in_channels, num_stages[0], bias=True)
        self.layer1 = self._resLayer(num_stages[1], dropout_rate)
        self.layer2 = self._resLayer(num_stages[2], dropout_rate)

        if self.norm == 'BN':
            self.bn = nn.BatchNorm2d(num_stages[2], momentum=0.9)

        self.fc = nn.Linear(num_stages[2], num_classes)

        #print(self.layer1)

    def _resLayer(self, out_planes, dropout_rate):
        layers = (LinearResidualBlock(self.in_planes,
                                    out_planes,
                                    dropout_rate=dropout_rate,norm=self.norm))
        self.in_planes = out_planes

        return layers

    def forward(self, input):
        x = self.lin1(input)
        x = self.layer1(x)
        x = self.layer2(x)

        if self.norm == 'BN':
            x = self.bn(x)

        x = F.relu(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net=LinearResNet(in_channels=768, num_classes = 10, widen_factor=10, norm='MSN')
    y = net(torch.randn(32,768)) # M x CHW

    print(y.size())