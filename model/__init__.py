import os

import torch
import torch.nn as nn

from importlib import import_module


class model:
    def __init__(self, args, ckp):
        print('Making model...')

        self.args = args
        self.ckp = ckp
        self.device = torch.device('cpu' if args.cpu_only else 'cuda')
        self.module = import_module('model.'+args.model.lower())
        self.model = self.module.make_model(args).to(self.device)

        if not args.cpu_only and self.args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        # if args.load_path != '.':
        #    self.load(ckp.log_dir, args.pre_train, args.cpu_only)

        if self.args.print_model:
            print(self.model)

    def get_model(self):
        if self.args.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, is_best=False):
        target = self.get_model()
        if not os.path.exists(os.path.join(apath, 'model')):
            os.makedirs(os.path.join(apath, 'model'))

        torch.save(target.state_dict(), os.path.join(apath, 'model', 'model_latest.pt'))
        if is_best:
            torch.save(target.state_dict(), os.path.join(apath, 'model', 'model_best.pt'))

    def load(self, apath, pre_train, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Load model from : {}'.format(pre_train))
            self.get_model().load_state_dict(torch.load(pre_train, **kwargs), strict=False)
        else:
            if self.args.test_only:
                print('Load model from : {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
                self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                                                 strict=False)
            else:
                print('Load model from : {}'.format(os.path.join(apath, 'model', 'model_best.pt')))
                self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_best.pt'), **kwargs),
                                                 strict=False)
