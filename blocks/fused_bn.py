import torch
import torch.nn as nn


# class FusedConvBatchNorm2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1,
#                  padding_mode='zeros', eps=1e-5,
#                  momentum=0.1, affine=True, track_running_stats=True):
#
#         super(FusedConvBatchNorm2d, self).__init__()
#
#         self.affine = affine
#         self.eps = eps
#         self.momentum = momentum
#         self.track_running_stats = track_running_stats
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                               stride, padding, dilation, groups, True, padding_mode)
#
#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
#             self.beta = nn.Parameter(torch.zeros(out_channels))  # bias
#
#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))
#
#     def forward(self, input):
#         self._check_input_dim(input)
#
#         x = input.transpose(0, 1).contiguous().view(input.size(1), -1)
#         print("input size", input.size(), x.size())
#
#         if self.training:
#             # Compute the mean and standard deviation
#             self.batch_mean = x.mean(1)[:, None, None].view([-1])
#             self.batch_std = x.std(1)[:None, None].view([1, -1, 1, 1])
#
#         conv_weights = self.conv.weight
#         conv_bias = self.conv.bias
#
#         # print("gamma size", self.gamma.size())
#         print("batch_mean size", self.batch_mean.size())
#
#         print("Weight sizes", self.conv.weight.size(), conv_weights.size())
#         print("bias sizes", self.conv.bias.size(), conv_bias.size())
#         print("test size", ((self.batch_mean).
#                             div(self.batch_std.view([-1])).mul(self.gamma.view([-1])).add(self.beta)).size())
#
#         if self.affine:
#             conv_weights = conv_weights.div(self.batch_std)
#             conv_weights = conv_weights.mul(self.gamma)
#
#             conv_bias = (conv_bias - self.batch_mean).div(self.batch_std.view([-1]))
#             conv_bias = conv_bias.mul(self.gamma.view([-1])).add(self.beta)
#
#         else:
#             conv_weights = conv_weights.div(self.batch_std)
#             conv_bias = (conv_bias - self.batch_mean).div(self.batch_std.view([-1]))
#
#
#         self.conv.weights = torch.nn.Parameter(conv_weights)
#         self.conv.bias = torch.nn.Parameter(conv_bias)
#         return self.conv(input)



