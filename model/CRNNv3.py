import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def make_model(args):
    return CRNNv3(args)


class CRNNv3(nn.Module):
    def __init__(self, args, max_seq_length=96):
        super(CRNNv3, self).__init__()
        self.args = args
        lexicon_size = len(args.dictionary)
        self.embed = nn.Embedding(lexicon_size, args.embed_size)
        self.lstm_text = nn.LSTM(args.embed_size, args.embed_size, args.num_layers,
                                 bidirectional=False, batch_first=True)
        self.lstm_image = nn.LSTM(2 * args.embed_size, args.hidden_size, args.num_layers,
                                  bidirectional=False, batch_first=True)
        self.linear = nn.Linear(args.hidden_size, lexicon_size)
        self.max_seq_length = max_seq_length
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')

    def forward(self, features, labels, lengths):
        labels = self.embed(labels)
        features = features.unsqueeze(1)
        zero = torch.zeros(features.size()[0], 1, self.args.embed_size).to(self.device)
        embeddings_rnn = torch.cat((zero, labels), 1)
        packed_rnn = pack_padded_sequence(embeddings_rnn, lengths, batch_first=True)
        intermediate, _ = self.lstm_text(packed_rnn)
        intermediate = pad_packed_sequence(intermediate, batch_first=True)
        embeddings_cnn = features.repeat(1, intermediate[1][0], 1)
        embeddings = torch.cat((intermediate[0], embeddings_cnn), 2)
        packed_cnn = pack_padded_sequence(embeddings, intermediate[1], batch_first=True)
        hiddens, _ = self.lstm_image(packed_cnn)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states_text=None, states_image=None):
        sampled_idx = []
        inputs = torch.zeros(1, 1, self.args.embed_size).to(self.device)
        for i in range(self.max_seq_length):
            intermediate, states_text = self.lstm_text(inputs, states_text)
            intermediate = torch.cat((intermediate, features.unsqueeze(1)), 2)
            hiddens, states_image = self.lstm_image(intermediate, states_image)
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
