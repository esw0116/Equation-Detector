import datetime
import os
import numpy as np
import pandas as pd
from scipy import misc
from functools import reduce

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision import utils
from torch.autograd import Variable

from model import model

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class logger:
    def __init__(self, args):
        def make_dir(base_dir):
            i = 1
            while 1:
                log_dir = os.path.join(base_dir, self.today + '_' + args.model + '_{:03}'.format(i))
                if os.path.exists(log_dir):
                    i = i + 1
                else:
                    os.makedirs(log_dir)
                    break
            return log_dir

        self.args = args
        self.today = datetime.datetime.now().strftime('%Y%m%d')
        if not args.test_only:
            self.log_dir = make_dir(args.log_dir)
            #self.log = torch.Tensor()
            print('Save Directory : {}'.format(self.log_dir))
            with open(self.log_dir + '/config.txt', 'w') as f:
                f.write(self.today + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            self.log_dir = args.load_path

    def load(self):
        my_model = model(self.args).get_model()
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())

        optimizer = optim.Adam
        kwargs = {'lr': self.args.learning_rate, 'weight_decay': 0}
        my_optimizer = optimizer(trainable, **kwargs)

        my_scheduler = lrs.StepLR(my_optimizer, step_size=self.args.lr_decay, gamma=self.args.gamma)

        my_model.load_state_dict(torch.load(self.log_dir + '/model/model_lastest.pt'))
        '''
        my_loss = torch.load(self.log_dir + '/loss.pt')
        my_optimizer.load_state_dict(
            torch.load(self.log_dir + '/optimizer.pt'))
        '''
        print('Load loss function from checkpoint...')

        return my_model, my_optimizer, my_scheduler

    def save(self, trainer, is_best=False):
        trainer.model.save(apath=self.log_dir, is_best=is_best)
        # trainer.loss.save(self.log_dir)
        # trainer.loss.plot_loss(self.log_dir, epoch)
        '''
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.log_dir, 'optimizer.pt')
        )
        '''

    def save_img(self, filename, img):
        if not os.path.exists(os.path.join(self.log_dir, 'result_img')):
            os.makedirs(os.path.join(self.log_dir, 'result_img'))
        if isinstance(img, Variable):
            utils.save_image(img.data.cpu(), os.path.join(self.log_dir, 'result_img/{}.png'.format(filename)), padding=0)
        elif isinstance(img, torch.Tensor):
            utils.save_image(img, os.path.join(self.log_dir, 'result_img/{}.png'.format(filename)), padding=0)
        elif isinstance(img, np.ndarray):
            if img.ndim == 4:
                img = np.transpose(np.squeeze(img, axis=0), axes=(1, 2, 0))
            if not img.dtype == 'uint8':
                img = (img*255).astype('uint8')
                img = np.clip(img, 0, 255)
            misc.imsave(os.path.join(self.log_dir, 'result_img/{}.png'.format(filename)), img)

    def plot(self, data, aux_data):
        def _init_figure(label):
            fig = plt.figure()
            plt.title(label)
            plt.xlabel('Iterations')
            plt.grid(True)
            return fig

        def _close_figure(fig, filename):
            plt.savefig(filename)
            plt.close(fig)

        def _smooth_data(data, smooth_length):
            smooth_data = data
            for i in range(smooth_length, len(data)-smooth_length):
                tmp_data = data[i-smooth_length:i+smooth_length+1]
                sum = reduce(lambda a, b: a+b, tmp_data)
                smooth_data[i] = sum / (1+2*smooth_length)
            return smooth_data

        axis = np.linspace(1, len(data), len(data))
        #label = '{} Loss'.format(loss['type'])
        label = 'Loss'
        fig = _init_figure(label)
        data = _smooth_data(data, 62)
        plt.plot(axis, data, label=label)
        for i in range(1, len(aux_data)):
            plt.axvline(aux_data[i], color='r')
        plt.legend()
        _close_figure(fig, '{}/loss.pdf'.format(self.log_dir))

        fig = _init_figure(label)
        plt.plot(axis, data, label=label)
        plt.legend()
        for i in range(1, len(aux_data)):
            plt.axvline(aux_data[i], color='r')
        plt.ylim(0, 5)
        _close_figure(fig, '{}/loss_magnified.pdf'.format(self.log_dir))

        '''
        set_name = type(trainer.loader_test.dataset).__name__
        fig = _init_figure('SR on {}'.format(set_name))
        for idx_scale, scale in enumerate(self.args.scale):
            legend = 'Scale {}'.format(scale)
            plt.plot(axis, test[:, idx_scale].numpy(), label=legend)
            plt.legend()

        _close_figure(
            fig,
            '{}/test_{}.pdf'.format(self.log_dir, set_name))
        '''
