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
        if args.load or args.test_only:
            self.load(args.load_path, args.cpu_only)

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

        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        self.get_model().load_state_dict(torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs), strict=False )
