import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

'''
Custom implementation of Inception V3 for input image of 96 * 480
Inception Paper: https://arxiv.org/pdf/1512.00567.pdf
'''

def make_model(args):
    return Inceptionv3(args)

'''
Network Structure:
Input = 96*480*1
Conv2d 1: 96*480*32
Conv2d 2: 96*480*64
Conv2d 4 (stride 2, padding 1): 48*240*64
Conv2d 3: 48*240*128
Inception A: 48*240*192
Conv2d 5 (stride 2, padding 1): 24*120*192
Inception B 1: 12*60*256
Inception B 2: 6*30*256
Conv2d 6 (stride 2, padding 1): 3*15*256
Conv2d 1x1: 3*15*128        Just to match 3*15*128 Output

-------------------------
FC 1: 5760 --> 400
FC 2: 400 --> 118
'''

class Inceptionv3(nn.Module):
    def __init__(self, args):
        super(Inceptionv3, self).__init__()
        # Convolutions
        self.Conv2d_1 = BasicConv(1, 32, kernel_size = 3, padding = 1)
        self.Conv2d_2 = BasicConv(32, 64, kernel_size = 3, padding = 1)
        self.Conv2d_3 = BasicConv(64, 128, kernel_size = 3, padding = 1)
        self.Conv2d_4 = BasicConv(64, 64, kernel_size = 3, padding = 1, stride = 2)
        self.Conv2d_5 = BasicConv(192, 192, kernel_size = 3, padding = 1, stride = 2)
        self.Conv2d_6 = BasicConv(256, 256, kernel_size = 3, padding = 1, stride = 2)
        self.Conv2d_1x1 = BasicConv(256,128, kernel_size =1)
        self.Max_pool = nn.MaxPool2d(2)
        #Inception Layers
        self.InceptionA = InceptionA(128)
        self.InceptionB_1 = InceptionB(192)
        self.InceptionB_2 = InceptionB(256)

        # Fully Connected
        self.fc1 = nn.Sequential(
            nn.Linear(5760, 400),
            nn.ReLU())
        self.fc2 = nn.Linear(400, 118)

    def forward(self, x):
        out = self.Conv2d_1(x)
        out = self.Conv2d_2(out)
        out = self.Conv2d_4(out)
        out = self.Conv2d_3(out)
        out = self.InceptionA(out)
        out = self.Conv2d_5(out)
        out = self.InceptionB_1(out)
        out = self.InceptionB_2(out)
        
        out = self.Conv2d_6(out)
        out = self.Conv2d_1x1(out)
        
        #FC
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def reset(self):
        for m in self.children():
            print(m)
            input()
            if isinstance(m, BasicConv) or isinstance(m, InceptionA) or isinstance(m, InceptionB):
                m.reset()

            if isinstance(m, nn.Conv2d):
                print('!')
                nn.init.kaiming_normal_(m.weight.data)

'''
Inception kernel which splits into 4 branches. Normally:
Branch 1: 1x1 --> 5x5 --> filter concat
Branch 2: 1x1 --> 3x3 --> filter concat
Branch 3: Pool --> 1x1 --> filter concat
Branch 4: 1x1 --> filter concat

All 5x5 convs are split into 3x3 and 3x3 convolutions due to them having same receptive field as 5x5
No dimension reduction in image
'''
class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv(in_channels, 64, kernel_size = 1)

        self.branch3x3_1 = BasicConv(in_channels, 32, kernel_size = 1)
        self.branch3x3_2 = BasicConv(32, 64, kernel_size = 3, padding = 1)
        self.branch3x3_3 = BasicConv(64, 64, kernel_size = 3, padding = 1)

        self.branch3x3 = BasicConv(in_channels, 32, kernel_size = 3, padding = 1)

        self.branch_pool = BasicConv(in_channels, 32, kernel_size = 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch3x3_1(x)
        branch5x5 = self.branch3x3_2(branch5x5)
        branch5x5 = self.branch3x3_3(branch5x5)

        branch3x3 = self.branch3x3(x)

        branch_pool = f.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]

        return torch.cat(outputs, 1)

    def reset(self):
        print('A')
        for m in self.children():
            if isinstance(m, BasicConv):
                m.reset()

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv(in_channels, 64, kernel_size = 3, stride = 2, padding = 1)
        
        self.branch1x1_3x3_1 = BasicConv(in_channels, 32, kernel_size = 1)
        self.branch1x1_3x3_2 = BasicConv(32, 64, kernel_size = 3, padding = 1)
        self.branch1x1_3x3_3 = BasicConv(64, 64, kernel_size = 3, stride = 2, padding = 1)

        self.branch_pool_conv = BasicConv(in_channels, 128, kernel_size =1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch1x1_3x3 = self.branch1x1_3x3_1(x)
        branch1x1_3x3 = self.branch1x1_3x3_2(branch1x1_3x3)
        branch1x1_3x3 = self.branch1x1_3x3_3(branch1x1_3x3)

        branch_pool = self.branch_pool_conv(x)
        branch_pool = self.max_pool(branch_pool)

        outputs = [branch3x3, branch1x1_3x3, branch_pool]
        return torch.cat(outputs, 1)

    def reset(self):
        print('B')
        for m in self.children():
            if isinstance(m, BasicConv):
                m.reset()

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x

    def reset(self):
        print('C')
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                print('!')
                nn.init.kaiming_normal_(m.weight.data)
