import torch.nn as nn

from model import model


def make_model(args):
    CRNN(args)

# input model name for model
class EncoderCNN(nn.Module):
    def __init__(self, model, ckp):
        super(EncoderCNN, self).__init__()
        my_model = model.make_model()
        my_model.reset()
        self.ckp = ckp
        
        my_model.load_state_dict(torch.load(os.path.join(ckp.log_dir, 'model', 
                'model_best.pt'), **kwargs), strict=False)
        
        # delete Fully Connected layer
        modules = list(my_model.childre())[:-1]
        self.my_model = nn.Sequential(*modules)
    
    def forward(self, x):
        with torch.no_grad():       ## remove if fine tuning
            features = self.my_model(x)
        #flatten for RNN Input
        features = features.reshape(features.size(0), -1)
        return features

class BidirectionalLSTM(nn.Module):

    def __init__(self, args, embed_size, hidden_size, lexicon_size, num_layers, max_seq_length=96):
        super(CRNN, self).__init__()
        self.embed = nn.Embedding(lexicon_size, embed_size)
        # num layers can be 1 or 2
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, batch_first = True)
        self.linear = nn.Linear(hidden_size, lexicon_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, labels, lengths):
        embeddings = self.embed(labels)
        # batch at dimension 0
        embeddings = torch.cat((features.unsqueeze(1), embeddings),1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first = True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states = None):
        sampled_idx = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_idx.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_idx = torch.stack(sampled_idx, 1)
        return sampled_idx

'''
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

'''
