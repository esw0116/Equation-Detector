import torch
import torch.nn as nn
from importlib import import_module


def make_model(args):
    return EncoderCNN(args)

class EncoderCNN(nn.Module):
    def __init__(self, args):
        super(EncoderCNN, self).__init__()
        self.args = args

        cnn_model = import_module('model.' + args.cnn_model)
        my_model = cnn_model.make_model(args)
        my_model.reset()

        # print("Loading CNN({}) Model!".format(args.cnn_model))
        # my_model.load_state_dict(torch.load('./CNN_Pretrained/{}.pt'.format(args.cnn_model)), strict=False)
        # print("Model Loaded!")

        # delete Fully Connected layer
        modules = list(my_model.children())[:-1]
        self.my_model = nn.Sequential(*modules)
        self.linear = nn.Linear(15 * 3 * 128, args.embed_size)
        self.bn = nn.BatchNorm1d(args.embed_size, momentum=0.01)

    def forward(self, x):
        if not self.args.fine_tune:
            with torch.no_grad():
                features = self.my_model(x)
        else:
            features = self.my_model(x)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features

    def reset(self):
        return