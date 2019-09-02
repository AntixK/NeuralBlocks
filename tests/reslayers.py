from NeuralBlocks.blocks.resblock import LinearResidualBlock
from NeuralBlocks.blocks.linnorm import LinarNorm

import torch

x = torch.randn(32, 768) # M x (CHW)
# net = LinarNorm(768, 10, norm='BN')
net = LinearResidualBlock(768, 10,norm='BN')
y = net(x)

print(y.size())
