import torch
import torch.nn as nn
from NeuralBlocks.blocks.convnormrelu import ConvNormRelu

class segnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, num_conv = 2):
        super(segnetDown, self).__init__()
        module = []

        if num_conv < 2:
            raise ValueError("SegNet needs at least 2 conv layers i.e. num_conv >= 2")

        """
        For SegNet, the down sampling layers have the form
            conv (in_channels, out_channels) + BN + ReLU
            conv (out_channels, out_channels) + BN + ReLU
        """
        num_filters= [in_channels] + (num_conv)*[out_channels]

        for i in range(num_conv):
            module.append(ConvNormRelu(num_filters[i], num_filters[i + 1],
                                       kernel_size=3, stride=1, padding=1, norm=norm))
        self.layer = nn.Sequential(*module)

        #print(self.layer)
        self.maxpool_argmax = nn.MaxPool2d(2,2, return_indices=True)

    def forward(self, input):
        output = self.layer(input)
        unpoolsed_size = output.size()
        output, indices = self.maxpool_argmax(output)
        return output, indices, unpoolsed_size


class segnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, is_deconv, num_conv = 2, norm = 'BN'):
        super(segnetUp, self).__init__()

        if num_conv < 2:
            raise ValueError("SegNet needs at least 2 conv layers i.e. num_conv >= 2")

        num_filters = [in_channels]*(num_conv) + [out_channels]
        """
        For SegNet, the up sampling layers have the form
            conv (in_channels, in_channels) + BN + ReLU
            conv (in_channels, out_channels) + BN + ReLU
        """

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.MaxUnpool2d(2,2)

        module = []
        for i in range(num_conv):
            module.append(ConvNormRelu(num_filters[i], num_filters[i + 1],
                                    kernel_size = 3, stride= 1, padding=1, norm=norm))
        self.layer = nn.Sequential(*module)

    def forward(self, input, indices, output_size):
        output = self.up(input = input, indices = indices, output_size=output_size)
        output =self.layer(output)

        return output

class SegNet(nn.Module):
    def __init__(self, in_channels, n_class, norm = 'BN', filters = None, is_deconv = False):

        super(SegNet,self).__init__()

        self.is_deconv = is_deconv

        if filters is None:
            filters = [64,128,256,512, 512]

        if len(filters) < 3:
            raise ValueError('Number filters must be at least 3.')

        filters.insert(0, in_channels) # To account for the initial channels

        modules= []
        # Downsampling phase
        for i in range(1, len(filters)):
            if i < 3:
                modules.append(segnetDown(filters[i-1], filters[i], norm, num_conv=2))
            else:
                modules.append(segnetDown(filters[i-1], filters[i], norm, num_conv=3))

        self.down_layers = nn.ModuleList(modules)

        # Upsampling Phase
        filters[0] = n_class # To account for the final number of classes
        modules = []
        for i in range(len(filters)-1,0,-1):
            if i > 2:
                modules.append(segnetUp(filters[i], filters[i-1], self.is_deconv,
                                        num_conv = 3, norm = norm))
            else:
                modules.append(segnetUp(filters[i], filters[i-1], self.is_deconv,
                                        num_conv = 2, norm = norm))

        self.up_layers = nn.ModuleList(modules)
        # print(self.up_layers)

    def forward(self, input):

        x = input
        unpool_args = []
        for i, module in enumerate(self.down_layers):
            x, ind, unpool_shape = module(x)
            unpool_args.append([ind, unpool_shape])

        result = x
        N = len(self.up_layers)-1 # Variable to traverse unpool_args from reverse

        """
        Note that the parameters for the up layers are the result of the
        previous layer and the unpool args from the corresponding up layer.
        i.e. the unpool_args must be traversed from reverse.
        """
        for i, module in enumerate(self.up_layers):
            result = module(result, *unpool_args[N-i])
        return result


if __name__ == "__main__":
    s = SegNet(3, 10, norm = 'BN')

    inp = torch.randn(32,3,128, 128) #M x C x H x W
    s.train()
    result = s(inp)
