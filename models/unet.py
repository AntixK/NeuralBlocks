import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralBlocks.blocks.convnorm import ConvNorm
from NeuralBlocks.blocks.meanspectralnorm import MeanSpectralNormConv2d

class unetDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None):
        super(unetDown, self).__init__()

        if norm is None:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 0),
                nn.ReLU(inplace=True))
        elif norm == 'MSN':
            self.layer = nn.Sequential(
                MeanSpectralNormConv2d(in_channels, out_channels, 3, 1, 0),
                nn.ReLU(inplace=True),
                MeanSpectralNormConv2d(out_channels, out_channels, 3, 1, 0),
                nn.ReLU(inplace=True))
        else:
            if norm != 'BN':
                raise UserWarning('Undefined normalization ' + norm + '. Using BatchNorm instead.')
            self.layer = nn.Sequential(
                        ConvNorm(in_channels, out_channels, norm, conv_args=(3, 1, 0, True)),
                        nn.ReLU(inplace=True),
                        ConvNorm(out_channels, out_channels, norm, conv_args=(3, 1, 0, True)),
                        nn.ReLU(inplace=True))

    def forward(self, input):
        return self.layer(input)

class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetDown(in_channels, out_channels, norm=None)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size(2) - input1.size(2)
        padding = 2*[int(offset/2), offset//2]
        #print(rint(offset/2), int(offset/2), padding)

        output1 = F.pad(input1, padding)
        merged_output = torch.cat([output1, output2],1)
        return self.conv(merged_output)

class UNet(nn.Module):
    def __init__(self, in_channels, n_class, norm = 'BN', filters = None, filter_scale = 4, is_deconv = True):

        super(UNet,self).__init__()

        self.is_deconv = is_deconv

        if filters is None:
            filters = [64,128,256,512,1024]

        if len(filters) < 3:
            raise ValueError('Number filters must be at least 3.')

        filters = [int(f/filter_scale) for f in filters]
        filters.insert(0, in_channels)

        modules= []
        # Downsampling phase
        for i in range(1, len(filters)-1):
            modules.append(unetDown(filters[i-1], filters[i], norm))
            modules.append(nn.MaxPool2d(kernel_size=2))

        modules.append(ConvNorm(filters[-2], filters[-1], 'BN', conv_args=(3,1,0, True)))

        self.down_layers = nn.ModuleList(modules)

        modules = []
        for i in range(len(filters)-1,1,-1):
            modules.append(unetUp(filters[i], filters[i-1], self.is_deconv))

        self.up_layers = nn.ModuleList(modules)
        self.final = nn.Conv2d(filters[1], n_class, 1)

    def forward(self, input):

        tensors = [input]
        for i, module in enumerate(self.down_layers):
            tensors.append(module(tensors[i]))

        result = tensors[-1]
        for i, module in enumerate(self.up_layers):
            result = module(tensors[-(2*i +3)], result)
        return self.final(result)


if __name__ == "__main__":
    u = UNet(3, 10, norm = 'MSN')

    inp = torch.randn(32,3,512,512) #M x C x H x W
    u.train()
    result = u(inp)













