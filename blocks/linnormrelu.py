import torch.nn as nn
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormLinear
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormLinReLU
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormLinear
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormLinReLU

class LinarNormReLU(nn.Module):

    def __init__(self,in_features, out_features, bias=True, norm='BN'):
        super(LinarNormReLU, self).__init__()

        if norm not in ['BN', 'IN','LN','WN', 'MSN', 'MSNTReLU', 'WNTRelU']:
            raise ValueError("Undefined norm value. Must be one of "
                             "['BN', 'IN','LN','WN', 'MSN', 'MSNTReLU', 'WNTRelU']")
        layers = []
        if norm == 'MSN':
            lin = MeanSpectralNormLinear(in_features, out_features, bias)
            layers += [lin, nn.ReLU(inplace=True)]
        elif norm == 'MSNTReLU':
            lin = MeanSpectralNormLinReLU(in_features, out_features, bias)
            layers += [lin]
        elif norm == 'WN':
            lin = MeanSpectralNormLinear(in_features, out_features, bias)
            layers += [lin, nn.ReLU(inplace=True)]
        elif norm == 'WNTReLU':
            lin = MeanWeightNormLinReLU(in_features, out_features, bias)
            layers += [lin]
        elif norm != 'IN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        elif norm != 'LN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.LayerNorm(out_features), nn.ReLU(inplace=True)]
        elif norm == 'BN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.BatchNorm2d(out_features), nn.ReLU(inplace=True)]
        else:
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.ReLU(inplace=True)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x