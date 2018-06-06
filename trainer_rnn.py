import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch import optim
import tqdm

from optimizer import set_optimizer, set_scheduler
from model import CRNN

DEBUG_MODE = False

class Trainer_RNN:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.model = model
        self.my_model = self.model.get_model()
        self.my_model.reset()
        self.loader_train, self.loader_test = loader
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')

        self.loss = nn.CrossEntropyLoss()
        self.encoder = CRNN.make_encoder(args).to(self.device)
        self.decoder = CRNN.make_decoder(args).to(self.device)

        # IF gradient only in RNN + linear layer:
        # self.params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
        # ELSE:
        self.params = list(self.decoder.parameters()) + list(self.encoder.parameters())

        # only train decoder
        self.optimizer = optim.Adam(self.params, args.learning_rate)
        self.lr_scheduler = set_scheduler(args, self.optimizer)

        if args.cpu_only:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if args.load_path != '.':
            if args.pre_train != '.':
                print('Load model from : {}'.format(args.pre_train))
                self.my_model.load_state_dict(torch.load(args.pre_train, **kwargs), strict=False)
            else:
                if not self.args.test_only:
                    print('Load model from : {}'.format(os.path.join(ckp.log_dir, 'model', 'model_latest.pt')))
                    self.my_model.load_state_dict(
                        torch.load(os.path.join(ckp.log_dir, 'model', 'model_latest.pt'), **kwargs),
                        strict=False)
                else:
                    print('Load model from : {}'.format(os.path.join(ckp.log_dir, 'model', 'model_best.pt')))
                    self.my_model.load_state_dict(
                        torch.load(os.path.join(ckp.log_dir, 'model', 'model_best.pt'), **kwargs),
                        strict=False)

            if not self.args.test_only:
                self.optimizer.load_state_dict(torch.load(os.path.join(ckp.log_dir, 'optimizer.pt')))
                for _ in range(1, len(ckp.loss.result)):
                    self.lr_scheduler.step()

        if args.CNN_pre != '.':
            print('Load CNN params...')
            self.my_model.cnn.load_state_dict(torch.load(args.CNN_pre, **kwargs), strict=False)
            print("Loaded CNN params!")

    def train(self):
        lr_before = self.lr_scheduler.get_lr()[0]
        self.lr_scheduler.step()
        self.my_model.train()
        lr_after = self.lr_scheduler.get_lr()[0]
        epoch = self.lr_scheduler.last_epoch + 1

        if lr_before != lr_after:
            self.ckp.loss.detect_lr_change(epoch)

        sum_loss = 0
        record_iter = len(self.loader_train) // 10
        tqdm_loader = tqdm.tqdm(self.loader_train)

        for idx, (_, img, capt, length) in enumerate(tqdm_loader):
            self.optimizer.zero_grad()

            images = img.to(torch.float).to(self.device)
            captions = capt.to(self.device) 
            targets = pack_padded_sequence(captions, length, batch_first=True)[0]

            features = self.encoder(images)
            outputs = self.decoder(features, captions, length)
            error = self.loss(outputs, targets)
            self.decoder.zero_grad()
            self.encoder.zero_grad()
            error.backward()
            self.optimizer.step()

            if DEBUG_MODE:
                print("Caption: ", captions.size())
                print("Target: ", targets)
                print("Output: ", outputs.argmax(dim=1))
                print("Chk: ", targets - outputs.argmax(dim=1))

            error = error.data.item()
            sum_loss += error

            if (idx + 1) % record_iter == 0:
                self.ckp.loss.register_loss(sum_loss/record_iter)
                sum_loss = 0

            tqdm_loader.set_description("CLoss: {:.4f}, LR: {:10.1e}".format(error, lr_after))

    def test(self):
        epoch = self.lr_scheduler.last_epoch + 1
        self.encoder.eval()
        self.decoder

        num_correct = 0
        fname_list = []
        table = np.zeros((3, len(self.loader_test)))

        tqdm_loader = tqdm.tqdm(self.loader_test)
        for idx, (fname, image, capt, length) in enumerate(tqdm_loader):
            images = image.to(torch.float).to(self.device)
            captions = capt.to(self.device)
            # with torch.autograd.no_grad():
            features = self.encoder(images)
            # output = self.decoder(features, captions, length)
            output = self.decoder.sample(features)
            fname_list.append(fname)
            # print("Output: ", output)
            # print("Ground Truth: ", capt)
            # table[0:2, idx] = [output, labels]
            cnt = 0
            if captions.size() == output.size():
                for i in range(captions.size(1)):
                    if captions[0][i] == output[0][i]:
                        cnt += 1
                if cnt == captions.size(1):
                    num_correct += 1
                    

        print('In Epoch {}, Acc is {}'.format(epoch, num_correct/len(self.loader_test)))
        self.ckp.save_results(fname_list, table)
        if not self.args.test_only:
            cur_best = torch.max(self.ckp.loss.result).item()
            self.ckp.loss.register_result(num_correct/len(self.loader_test))
            self.ckp.save(self, self.my_model, epoch, is_best=num_correct/len(self.loader_test) > cur_best)

    def termination(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.lr_scheduler.last_epoch + 1
            return epoch >= self.args.num_epochs
