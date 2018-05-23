import os
import glob
from importlib import import_module
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import Character


class data:
    def __init__(self, args):
        self.args = args
        self.kwargs = {}
        if not args.cpu_only:
            self.kwargs['num_workers'] = 0
            self.kwargs['pin_memory'] = True

    def get_loader(self):
        module = import_module('data.' + self.args.work_type)
        if self.args.work_type == 'Character':
            trainset = getattr(module, self.args.work_type)(self.args)
            loader_train = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            testset = getattr(module, self.args.work_type)(self.args, train=False)
            loader_test = DataLoader(testset, batch_size=1, shuffle=False, **self.kwargs)

            return loader_train, loader_test
