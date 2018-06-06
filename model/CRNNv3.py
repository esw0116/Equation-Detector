import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def make_model(args):
    return CRNNv2(args)


class CRNNv2(nn.Module):
    def __init__(self, args, max_seq_length=96):
        super(CRNNv2, self).__init__()
        self.args = args
        lexicon_size = len(args.dictionary)
        self.embed = nn.Embedding(lexicon_size, args.embed_size)
        self.lstm = nn.LSTM(2 * args.embed_size, args.hidden_size, args.num_layers, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(args.hidden_size, lexicon_size)
        self.max_seq_length = max_seq_length
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')

    def forward(self, features, labels, lengths):
        labels = self.embed(labels)
        features = features.unsqueeze(1)
        embeddings_cnn = features.repeat(1, lengths[0]+1, 1)
        zero = torch.zeros(features.size()[0], 1, self.args.embed_size).to(self.device)
        embeddings_rnn = torch.cat((zero, labels), 1)
        embeddings = torch.cat((embeddings_cnn, embeddings_rnn), 2)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        sampled_idx = []
        zero = torch.zeros(1, 1, self.args.embed_size).to(self.device)
        inputs = torch.cat((features.unsqueeze(1), zero), 2)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_idx.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            inputs = torch.cat((features.unsqueeze(1), inputs), 2)
            if predicted == 2:
                break
        sampled_idx = torch.stack(sampled_idx, 1)
        return sampled_idx

    def reset(self):
        return
