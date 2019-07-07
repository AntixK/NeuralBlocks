import torch

def channelShuffle(input, groups=16):
    M, C, H, W = input.data.size()

    channels_per_group = C//groups

    input = input.view(M, groups, channels_per_group, H, W)
    print(input.size())

    #- contiguous() required if transpose() is used before view().
    input = torch.transpose(input,1,2).contiguous()
    print(input.size())

    # Flatten
    input = input.view(M, -1, H,W)
    return input

x = torch.randn(1,2,3,3)
print(x)
print(channelShuffle(x, 2))