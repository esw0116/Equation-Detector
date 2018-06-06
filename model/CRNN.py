import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from importlib import import_module


def make_model(args):
    return CRNN(args)


class CRNN(nn.Module):
    def __init__(self, args, max_seq_length=96):
        super(CRNN, self).__init__()
        self.args = args
        lexicon_size = len(args.dictionary)
        self.embed = nn.Embedding(lexicon_size, args.embed_size)
        self.lstm = nn.LSTM(args.embed_size, args.hidden_size, args.num_layers, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(args.hidden_size, lexicon_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, labels, lengths):
        embeddings = self.embed(labels)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        sampled_idx = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_idx.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted == 2:
                break
        sampled_idx = torch.stack(sampled_idx, 1)
        return sampled_idx

    def reset(self):
        return
