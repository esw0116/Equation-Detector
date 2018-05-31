import os
import numpy as np
import torch
import torch.nn as nn
import tqdm

from optimizer import set_optimizer, set_scheduler


class Trainer_CNN:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.model = model
        self.my_model = self.model.get_model()
        self.loader_train, self.loader_test = loader
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')

        self.optimizer = set_optimizer(args, self.my_model)
        self.lr_scheduler = set_scheduler(args, self.optimizer)
        if args.load_path != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.log_dir, 'optimizer.pt')))
            for _ in range(len(self.ckp.loss.log)):
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

        sum_loss = torch.zeros(1)
        tqdm_loader = tqdm.tqdm(self.loader_train)

        for idx, (img, label) in enumerate(tqdm_loader):
            self.optimizer.zero_grad()

            images = img.to(torch.float).to(self.device)
            labels = label.to(torch.long).to(self.device)
            output = self.my_model(images)

            error = self.loss(output, labels)
            error.backward()
            self.optimizer.step()

            error = error.data.item()
            sum_loss += error

            tqdm_loader.set_description("CLoss: {:.4f}, LR: {:10.1e}".format(error, lr_after))

        self.ckp.loss.register_loss(sum_loss.div(len(self.loader_train)))

    def test(self):
        epoch = self.lr_scheduler.last_epoch + 1
        self.my_model.eval()
        num_correct = np.zeros(1)
        fname_list = []
        table = np.zeros((3, len(self.loader_test)))
        for idx, (fname, image, label) in enumerate(self.loader_test):
            images = image.to(torch.float).to(self.device)
            labels = label.to(torch.long).to(self.device)
            with torch.autograd.no_grad():
                output = self.my_model(images)
            fname_list.append(fname)
            table[idx, 0:2] = [output.argmax(), labels]
            if labels == output.argmax():
                num_correct += 1
                table[idx, 2] = 1

        print('In Epoch {}, Acc is {}'.format(epoch, num_correct/len(self.loader_test)))
        cur_best = torch.max(self.ckp.loss.result).item()
        self.ckp.save(self, epoch, is_best=num_correct/len(self.loader_test) > cur_best)
        self.ckp.loss.register_result(torch.from_numpy(num_correct).div(len(self.loader_test)))

        f.write('Test Accuracy : {:.4f}'.format(num_correct/len(self.loader_test)))

    def termination(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.lr_scheduler.last_epoch + 1
            return epoch >= self.args.num_epochs
