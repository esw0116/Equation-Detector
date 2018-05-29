import torch
import torch.nn as nn

from model import common


def make_model(args):
    return baseline(args)


class baseline(nn.Module):
    def __init__(self, args, conv = common.default_conv):
        super(baseline, self).__init__()
        # TODO : Change Depth

        conv_list = []
        conv_list.extend([conv(1, 16, 3), nn.ReLU(True)])
        conv_list.extend([conv(16, 16, 3), nn.ReLU(True)])
        conv_list.append(nn.MaxPool2d(2))
        conv_list.extend([conv(16, 32, 3), nn.ReLU(True)])
        conv_list.extend([conv(32, 32, 3), nn.ReLU(True)])
        conv_list.append(nn.MaxPool2d(2))
        conv_list.extend([conv(32, 64, 3), nn.ReLU(True)])
        conv_list.extend([conv(64, 64, 3), nn.ReLU(True)])
        conv_list.append(nn.MaxPool2d(2))
        conv_list.extend([conv(64, 128, 3), nn.ReLU(True)])
        conv_list.extend([conv(128, 128, 3), nn.ReLU(True)])
        conv_list.append(nn.MaxPool2d(2))
        conv_list.extend([conv(128, 128, 3), nn.ReLU(True)])
        conv_list.extend([conv(128, 128, 3), nn.ReLU(True)])
        conv_list.append(nn.MaxPool2d(2))
        conv_list.extend([conv(128, 128, 3), nn.ReLU(True)])

        fc_list = []
        fc_list.extend([nn.Linear(128*3*15, 400), nn.ReLU(True)])
        fc_list.append(nn.Linear(400, 82))

        self.body_conv = nn.Sequential(*conv_list)
        self.body_fc = nn.Sequential(*fc_list)

    def forward(self, x):
        x = self.body_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.body_fc(x)

        return x

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
