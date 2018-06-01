import os
import imageio
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms

from data import common


class Character(data.Dataset):
    def __init__(self, args, train=True):
        def _get_class_index(symbols, class_list):
            labels_list = []
            for filepath in symbols:
                class_name = os.path.basename(os.path.dirname(filepath))
                if class_name == 'dot':
                    class_name = '.'
                elif class_name == 'forward_slash':
                    class_name = '/'
                labels_list.append(class_list.index(class_name))
            return labels_list
        self.args = args
        self.train = train
        self.apath = os.path.join(self.args.data_path, 'Symbol')
        self.symbols_list = sorted(glob.glob(os.path.join(self.apath, '**/*.jpg'), recursive=True))
        self.labels_list = _get_class_index(self.symbols_list, self.args.dictionary)
        self.symbol_train, self.symbol_test, self.label_train, self.label_test = \
            train_test_split(self.symbols_list, self.labels_list, test_size=0.1, random_state=self.args.seed)
        # self.label_train = torch.from_numpy(np.asarray(self.label_train, dtype='uint8'))
        # self.label_test = torch.from_numpy(np.asarray(self.label_test, dtype='uint8'))

    def __getitem__(self, idx):
        if not self.train:
            image = imageio.imread(self.symbol_test[idx])
            image = common.preprocess(image)
            image = common.normalize_img(image)
            image = common.rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            label = self.label_test[idx]
            filename = self.symbol_test[idx]
            return filename, image, label

        else:
            image = imageio.imread(self.symbol_train[idx])
            image = common.preprocess(image)
            image = common.rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            label = self.label_train[idx]
            return image, label

    def __len__(self):
        if not self.train:
            return len(self.symbol_test)
        else:
            return len(self.symbol_train)
