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
                labels_list.append(class_list.index(class_name))
            return labels_list
        self.args = args
        self.train = train
        self.apath = os.path.join(self.args.data_path, self.args.data_name)
        self.symbols_list = sorted(glob.glob(os.path.join(self.apath, '**/*.jpg'), recursive=True))
        self.class_list = ([_path.split('/')[-2] for _path in glob.glob(os.path.join(self.apath, '*/'))])
        assert len(self.class_list) == 82
        self.labels_list = _get_class_index(self.symbols_list, self.class_list)
        self.symbol_train, self.symbol_test, self.label_train, self.label_test = \
            train_test_split(self.symbols_list, self.labels_list, test_size=0.1, random_state=self.args.seed)
        self.class_dict = dict()
        for i in range(len(self.class_list)):
            temp = np.zeros((len(self.class_list)))
            temp[i] = 1
            self.class_dict[i] = temp
        #print(self.class_dict)

    def __getitem__(self, idx):
        if not self.train:
            image = imageio.imread(self.symbol_test[idx])
            image = common.preprocess(image)
            image = common.rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            label = torch.from_numpy(self.class_dict[self.label_test[idx]])
            return image, label

        else:
            idx = idx % len(self.symbol_train)
            image = imageio.imread(self.symbol_train[idx])
            image = common.preprocess(image)
            image = common.rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            #print(self.label_train[idx])
            label = self.label_train[idx]
            #label = torch.from_numpy(self.class_dict[self.label_train[idx]]).type('torch.LongTensor')
            return image, label

    def __len__(self):
        if not self.train:
            return len(self.symbol_test)
        else:
            return self.args.batch_size * self.args.num_batches
