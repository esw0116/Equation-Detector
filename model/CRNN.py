import torch.nn as nn

from model import baseline
from model import common


def make_model(args):
    CRNN(args)


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        # T: sequence length, b: batch size,
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(baseline.baseline, BidirectionalLSTM):

    def __init__(self, args, conv=common.default_conv):
        super(CRNN, self).__init__(args, conv)
        self.cnn = nn.Sequential(*list(baseline.baseline.features.children())[0:-2])
        self.lstm = BidirectionalLSTM.features.children()

    def forward(self, x):
        cnn = self.cnn(x)
        B, _, _, _ = cnn.size()
        cnn = cnn.view(96, B, -1)
        # LSTM input shape (Sequence length, Batch size, input vector size)
        rnn = self.lstm(cnn)

        return rnn
