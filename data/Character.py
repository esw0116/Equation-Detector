import os
import imageio
import glob
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data


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
        self.apath = os.path.join(self.args.dir_data, self.args.data_name)
        self.symbols_list = sorted(glob.glob(os.path.join(self.apath, '**/*.png'), recursive=True))
        self.class_list = sorted(glob.glob(os.path.join(self.apath, '*/')))
        assert len(self.class_list) == 82
        self.labels_list = _get_class_index(self.symbols_list, self.class_list)

        self.symbol_train, self.symbol_test, self.label_train, self.label_test = \
            train_test_split(self.symbols_list, self.labels_list, test_size=0.1, random_state=self.args.seed)

    def __getitem__(self, idx):
        if not self.train:
            image = imageio.imread()
            # fname, _ = os.path.splitext(os.path.split(self.int_list_test[idx])[-1])
            fname, _ = os.path.splitext(os.path.split(self.int_list_train[idx+self.split])[-1])
            return [common.np2Tensor([a, b, c, d, e], self.args.rgb_range), fname]

        else:
            idx = idx % len(self.symbol_train)
            image = imageio.imread(self.symbol_train[idx])


    def __len__(self):
        if not self.train:
            return self.list_len_train - self.split
        else:
            return self.args.batch_size * self.args.num_batches