from torch import nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
    conv.weight.data.normal_(0, 0.001)
    conv.bias.data.fill_(0)
    return conv
