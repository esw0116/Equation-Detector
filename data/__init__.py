import os
import glob
import torch
from importlib import import_module
from torch.utils.data import DataLoader

from data import Character, Expression


class data:
    def __init__(self, args):
        self.args = args
        self.kwargs = {}
        if not args.cpu_only:
            self.kwargs['num_workers'] = 0
            self.kwargs['pin_memory'] = True

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[2]), reverse=True)
        filenames, images, captions = zip(*data)
        images = torch.stack(images, 0)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return filenames, images, targets, lengths

    def get_loader(self):
        module = import_module('data.' + self.args.work_type)
        if self.args.work_type == 'Character':
            trainset = getattr(module, self.args.work_type)(self.args)
            loader_train = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            testset = getattr(module, self.args.work_type)(self.args, train=False)
            loader_test = DataLoader(testset, batch_size=1, shuffle=False, **self.kwargs)

            return loader_train, loader_test
        
        if self.args.work_type == 'Expression':
            trainset = getattr(module, self.args.work_type)(self.args)
            loader_train = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True,
                                      collate_fn=self.collate_fn, **self.kwargs)
            testset = getattr(module, self.args.work_type)(self.args, train=False)
            loader_test = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=self.collate_fn, **self.kwargs)

            return loader_train, loader_test
