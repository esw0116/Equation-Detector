import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import os

from model import resnet
from model import baseline


def make_model(args):
    return CRNN(args, 512, 512, 122, 2)

def make_decoder(args):
    return CRNN(args, 512, 512, 122, 2)

def make_encoder(args):
    return EncoderCNN(args)


# input model name for model
class EncoderCNN(nn.Module):
    def __init__(self, args):
        super(EncoderCNN, self).__init__()
        my_model = baseline.make_model(args)
        my_model.reset()
        
        print("Loading Model!")
        my_model.load_state_dict(torch.load('./experiment/20180603_baseline_001/model/model_best.pt'), strict=False)
        # print("Parameters: ", my_model.parameters.data)
        # for param in my_model.parameters():
        #     print(param.data)
        print("Model Loaded!")  
        
        # delete Fully Connected layer
        modules = list(my_model.children())[:-1]


        self.my_model = nn.Sequential(*modules)
        # for params in self.my_model.parameters():
        #     print(params.data)
        self.linear = nn.Linear(15*3*128, 512)
        self.bn = nn.BatchNorm1d(512, momentum = 0.01)
    
    def forward(self, x):
        # with torch.no_grad():       ## remove if fine tuning
        features = self.my_model(x)
        #flatten for RNN Input
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    def reset():
        return

class CRNN(nn.Module):

    def __init__(self, args, embed_size, hidden_size, lexicon_size, num_layers, max_seq_length=96):
        super(CRNN, self).__init__()
        self.embed = nn.Embedding(lexicon_size, embed_size)
        # num layers can be 1 or 2
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=False, batch_first = True)
        self.linear = nn.Linear(hidden_size, lexicon_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, labels, lengths):
        embeddings = self.embed(labels)
        # print(embeddings)
        # print(embeddings.size())
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # print(embeddings)
        # print(input())
        # print(embeddings.size())
        packed = pack_padded_sequence(embeddings, lengths, batch_first = True)
        hiddens, _ = self.bilstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        sampled_idx = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.bilstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_idx.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted==2:
                break
        sampled_idx = torch.stack(sampled_idx, 1)
        return sampled_idx

    def reset(self):
        return
