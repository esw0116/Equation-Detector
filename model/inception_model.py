import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Inceptionv3(nn.Module):
    def __init__(self):
        super(Inceptionv3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
