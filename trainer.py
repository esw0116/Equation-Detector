import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import tqdm

from optimizer import set_optimizer, set_scheduler


class Trainer_CNN:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.model = model
        self.my_model = self.model.get_model()

        self.my_model.reset()
        self.loader_train, self.loader_test = loader
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')
        self.optimizer = optim.Adam(self.my_model.parameters(), args.learning_rate)
        # set_optimizer(args, self.my_model)
        self.lr_scheduler = set_scheduler(args, self.optimizer)

        if args.load_path != '.':
            if args.cpu_only:
                kwargs = {'map_location': lambda storage, loc: storage}
            else:
                kwargs = {}

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
                        strict=True)

            if not self.args.test_only:
                self.optimizer.load_state_dict(torch.load(os.path.join(ckp.log_dir, 'optimizer.pt')))
                for _ in range(1, len(ckp.loss.result)):
                    self.lr_scheduler.step()

        self.loss = nn.CrossEntropyLoss()

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

        for idx, (img, label) in enumerate(tqdm_loader):
            self.optimizer.zero_grad() 

            images = img.to(torch.float).to(self.device)
            labels = label.to(self.device)
            output = self.my_model(images)

            # print(labels[0])
            # print(output[0])

            error = self.loss(output, labels)
            error.backward()
            self.optimizer.step()

            error = error.data.item()
            sum_loss += error

            if (idx + 1) % record_iter == 0:
                self.ckp.loss.register_loss(sum_loss/record_iter)
                sum_loss = 0 

            tqdm_loader.set_description("CLoss: {:.4f}, LR: {:10.1e}".format(error, lr_after))


    def test(self):
        epoch = self.lr_scheduler.last_epoch + 1
        self.my_model.eval()

        num_correct = 0
        fname_list = []
        table = np.zeros((3, len(self.loader_test)))
        correct_dic = {'Filename': [], 'GroundTruth': [], 'Prediction': [], 'Correct': []}

        tqdm_loader = tqdm.tqdm(self.loader_test)
        for idx, (fname, image, label) in enumerate(tqdm_loader):
            images = image.to(torch.float).to(self.device)
            labels = label.to(self.device)
            # print(label)
            with torch.autograd.no_grad():
                output = self.my_model(images)
            fname_list.append(fname)
            table[0:2, idx] = [output.argmax(), labels]
            correct_dic['Filename'].append(fname)
            correct_dic['GroundTruth'].append(labels.data.cpu().numpy())
            correct_dic['Prediction'].append(output.argmax().data.cpu().numpy())
            if labels == output.argmax():
                num_correct += 1
                correct_dic['Correct'].append(1)
            else:
                correct_dic['Correct'].append(0)

        print('In Epoch {}, Acc is {}'.format(epoch, num_correct/len(self.loader_test)))
        self.ckp.save_results(correct_dic)
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
