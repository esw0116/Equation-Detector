import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels, stride, **kwargs)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BasicConv(out_channels, out_channels, stride, **kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride


        