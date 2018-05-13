import os

import torch
import torch.nn as nn

from importlib import import_module


class model:
    def __init__(self, args, ckp):
        self.args = args
        self.module = import_module('model.'+args.model)
        print('Making model...')
        self.model = self.module.make_model(args)
        if not self.args.no_cuda:
            print('CUDA is ready!')
            torch.cuda.manual_seed(args.seed)
            self.model.cuda()

            if self.args.n_GPUs > 1:
                self.model = nn.DataParallel(self.model, range(0, args.n_GPUs))

        self.load(ckp.log_dir, args.pre_train, args.resume, args.no_cuda)

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

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        else:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False)
