import os
import glob
from importlib import import_module
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import Character


class data:
    def __init__(self, args):
        def _get_class_index(symbols, class_list):
            labels_list = []
            for filepath in symbols:
                class_name = os.path.basename(os.path.dirname(filepath))
                labels_list.append(class_list.index(class_name))
            return labels_list

        self.args = args
        '''
        self.apath = os.path.join(self.args.data_path, self.args.data_name)
        self.symbols_list = sorted(glob.glob(os.path.join(self.apath, '**/*.jpg'), recursive=True))
        print(len(self.symbols_list))
        self.class_list = sorted(glob.glob(os.path.join(self.apath, '*/')))
        assert len(self.class_list) == 82
        self.labels_list = _get_class_index(self.symbols_list, self.class_list)

        self.symbol_train, self.symbol_test, self.label_train, self.label_test = \
            train_test_split(self.symbols_list, self.labels_list, test_size=0.1, random_state=self.args.seed)
        '''
        self.kwargs = {}
        if not args.no_cuda:
            self.kwargs['num_workers'] = 10
            self.kwargs['pin_memory'] = True

    def get_loader(self):
        module = import_module('data.' + self.args.work_type)
        if self.args.work_type == 'Character':
            trainset = getattr(module, self.args.work_type)(self.args)
            loader_train = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            testset = getattr(module, self.args.work_type)(self.args, train=False)
            loader_test = DataLoader(testset, batch_size=1, shuffle=False, **self.kwargs)

            return loader_train, loader_test
