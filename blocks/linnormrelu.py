import torch.nn as nn
from NeuralBlocks.blocks.linnorm import LinarNorm
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormLinReLU
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormLinReLU

class LinarNormReLU(nn.Module):

    def __init__(self,in_features, out_features, bias=True, norm='BN'):
        super(LinarNormReLU, self).__init__()

        if norm not in ['BN', 'IN','LN','WN', 'MSN', 'MWN', 'MSNTReLU', 'WNTRelU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "['BN', 'IN','LN','WN', 'MSN', 'MWN', 'MSNTReLU', 'WNTRelU']")
        layers = []
        if norm == 'MSNTReLU':
            lin = MeanSpectralNormLinReLU(in_features, out_features, bias)
            layers += [lin]
        elif norm == 'WNTReLU':
            lin = MeanWeightNormLinReLU(in_features, out_features, bias)
            layers += [lin]
        else:
            lin = LinarNorm(in_features, out_features, bias, norm=norm)
            layers += [lin, nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x