import datetime
import os
import numpy as np
import pandas as pd
from scipy import misc
from functools import reduce

import torch
from torchvision import utils
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class logger:
    def __init__(self, args):
        def make_dir(base_dir):
            i = 1
            while 1:
                log_dir = os.path.join(base_dir, self.today + '_' + args.model + '_{:03}'.format(i))
                if os.path.exists(os.path.join(log_dir, 'model')):
                    i = i + 1
                else:
                    if os.path.exists(log_dir):
                        print('Delete Incomplete Directory...')
                        os.system('rm -rf ' + log_dir)
                    os.makedirs(log_dir)
                    break
            return log_dir

        self.args = args
        self.loss = self.Loss()
        self.today = datetime.datetime.now().strftime('%Y%m%d')

        if args.load_path != '.':
            self.log_dir = os.path.join(args.log_dir, args.load_path)
            if os.path.exists(os.path.join(self.log_dir, 'model')):
                print('Load Directory : {}'.format(self.log_dir))
                self.loss.load(self.log_dir)
                print('Current Epoch : {}'.format(len(self.loss.result)-1))

            else:
                args.load_path = '.'

        if args.load_path == '.':
            self.log_dir = make_dir(args.log_dir)

            print('Save Directory : {}'.format(self.log_dir))
            with open(self.log_dir + '/config.txt', 'w') as f:
                f.write(self.today + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

    def save(self, trainer, model, epoch, is_best=False):
        # trainer.model.save(apath=self.log_dir, model, is_best=is_best)
        self.loss.save(self.log_dir)
        self.loss.plot_loss(self.log_dir, epoch)
        self.loss.plot_acc(self.log_dir, epoch)
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.log_dir, 'optimizer.pt'))
        if not os.path.exists(os.path.join(self.log_dir, 'model')):
            os.makedirs(os.path.join(self.log_dir, 'model'))
        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model', 'model_latest.pt'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(self.log_dir, 'model', 'model_best.pt'))

    def save_results(self, fname, array):
        column = ['Testdata', 'Prediction', 'Ground Truth', 'Correct?']
        df = pd.DataFrame(columns=column)
        df['Testdata'] = fname
        df['Prediction'] = array[0]
        df['Ground Truth'] = array[1]
        df['Correct?'] = array[2]

        df.to_csv(os.path.join(self.log_dir, 'result.csv'))

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

    class Loss:
        def __init__(self):
            self.log = torch.Tensor()
            self.lr_ch = []
            self.result = torch.zeros(1).to(torch.float)

        def load(self, apath):
            self.log = torch.load('{}/loss_log.pt'.format(apath))
            self.result = torch.load('{}/result.pt'.format(apath))

        def save(self, apath):
            torch.save(self.log, '{}/loss_log.pt'.format(apath))
            torch.save(self.result, '{}/result.pt'.format(apath))

        def register_loss(self, value):
            self.log = torch.cat((self.log, torch.zeros(1)))
            self.log[-1] = value

        def register_result(self, value):
            self.result = torch.cat((self.result, torch.zeros(1)))
            self.log[-1] = value

        def detect_lr_change(self, epoch):
            self.lr_ch.append(epoch)

        def plot_loss(self, apath, epoch):
            axis = np.linspace(1, epoch, epoch*10)
            label = 'Loss_Graph'
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log.numpy(), label=label)
            for i in range(1, len(self.lr_ch)):
                plt.axvline(self.lr_ch[i], color='r')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss.pdf'.format(apath))
            plt.close(fig)

        def plot_acc(self, apath, epoch):
            axis = np.linspace(1, epoch, epoch)
            label = 'Accuracy_Graph'
            fig = plt.figure()
            plt.title(label)
            data = self.result.numpy()
            plt.plot(axis, data[1:], label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig('{}/result.pdf'.format(apath))
            plt.close(fig)
