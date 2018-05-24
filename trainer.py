import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class Trainer:
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.model = model
        self.my_model = self.model.get_model()
        self.loader_train, self.loader_test = loader
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')

        self.optimizer = optim.Adam(self.model.get_model().parameters(), lr=self.args.learning_rate)
        self.adaptive_optim = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay_step,
                                                        gamma=self.args.gamma)

    def train(self):

        loss = nn.CrossEntropyLoss()

        self.my_model.reset()
        self.my_model.train()

        avg_loss = 0
        tqdm_loader = tqdm.tqdm(self.loader_train)
        loss_list = []
        lr_change = []
        learning_rate = 0

        for idx, (img, label) in enumerate(tqdm_loader):
            self.optimizer.zero_grad()

            images = img.to(torch.float).to(self.device)
            labels = label.to(self.device)
            output = self.my_model(images)

            error = loss(output, labels)
            error.backward()
            self.ckp.save(self, idx)
            self.adaptive_optim.step()
            self.optimizer.step()

            for params in self.optimizer.param_groups:
                if learning_rate != float(params['lr']):
                    learning_rate = float(params['lr'])
                    lr_change.append(idx)

            error = error.data.item()
            loss_list.append(error)
            avg_loss += error

            tqdm_loader.set_description("CLoss: {:.4f}, LR: {:10.1e}".format(error, learning_rate))

        self.ckp.plot(loss_list, lr_change)

    def test(self):
        self.my_model.eval()

        num_correct = 0
        for idx, (image, label) in enumerate(self.loader_test):
            images = image.to(torch.float).to(self.device)
            labels = label.to(self.device)
            with torch.autograd.no_grad():
                output = self.my_model(images)

            if labels.argmax() == output.argmax():
                num_correct += 1

        print(num_correct/idx)
