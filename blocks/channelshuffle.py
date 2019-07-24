import torch.nn as nn

class ChannelShuffle(nn.Module):
    """
    Channel Shuffle
    [M,C,H,W] -> [M, g, C/g, H,W] -> [M, C/g, g, H, W] -> [M, C, H, W]
    """

    def __init__(self, groups=16):
        super(ChannelShuffle, self).__init__()

        self.groups = groups

    def forward(self, input):

        M, C, H, W = input.data.size()

        channels_per_group = C//self.groups

        input = input.view(M, self.groups, channels_per_group, H, W)
        # print(input.size())

        #- contiguous() required if transpose() is used before view().
        input = torch.transpose(input,1,2).contiguous()
        # print(input.size())

        # Flatten
        input = input.view(M, -1, H,W)
        return input

if __name__ == '__main__':
    import torch
    x = torch.randn(1,2,3,3)
    # print(x)
    net = ChannelShuffle(groups=2)
    print(net(x).size())