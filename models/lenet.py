import torch.nn as nn
import torch.nn.functional as F
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class LeNet(nn.Module):
    def __init__(self, in_channels, num_class, norm='BN'):
        super(LeNet, self).__init__()
        self.conv1 = ConvNormRelu(in_channels, out_channels=20, kernel_size=5)
        self.conv2 = ConvNormRelu(20, out_channels=50, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 50, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    import torch
    input = torch.rand(32, 1, 28, 28)  # M x C x H x W
    net  =LeNet(1,10)
    print(net(input).size())