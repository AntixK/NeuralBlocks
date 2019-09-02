import torch.nn as nn
from NeuralBlocks.blocks.meanweightnorm import MeanWeightNormLinear
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormLinear

class LinarNorm(nn.Module):

    def __init__(self,in_features, out_features, bias=True, norm='BN'):
        super(LinarNorm, self).__init__()

        if norm not in ['BN', 'IN','LN','WN', 'MSN', 'MWN']:
            raise ValueError("Undefined norm value. Must be one of "
                             "['BN', 'IN','LN','WN', 'MSN', 'MWN']")
        layers = []
        if norm == 'MSN':
            lin = MeanSpectralNormLinear(in_features, out_features, bias)
            layers += [lin]
        elif norm == 'WN':
            lin = MeanSpectralNormLinear(in_features, out_features, bias)
            layers += [lin]
        elif norm == 'MWN':
            lin = MeanWeightNormLinear(in_features, out_features, bias)
            layers += [lin]
        elif norm == 'IN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.InstanceNorm2d(out_features)]
        elif norm != 'LN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.LayerNorm(out_features)]
        elif norm == 'BN':
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin, nn.BatchNorm2d(out_features)]
        else:
            lin = nn.Linear(in_features, out_features, bias)
            layers += [lin]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x