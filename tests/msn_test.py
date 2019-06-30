import torch
from torch import nn
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormLinear

class MSNConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MSNConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            MeanSpectralNormConv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            MeanSpectralNormConv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.fc = MeanSpectralNormLinear(32 * 28 * 28, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    input = torch.rand(32, 1, 28, 28)  # M x C x H x W
    c = input.size()
    #print(c)
    model = MSNConvNet()
    y = model(input)
    print(y.size())