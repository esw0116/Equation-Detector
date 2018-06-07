import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


def make_model(args):
    return ResNet(BasicBlock, [3, 4, 6, 3])


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

# Basic Resnet Block
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = BasicConv(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride = 2)
        self.conv2 = BasicConv(512, 128, kernel_size = 1)
        self.fc1 = nn.Linear(5760, 118)
        
    
    def make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes !=planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
    
    def reset(self):
        return