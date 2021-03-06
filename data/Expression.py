import os
import imageio
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd

from data import common

# csv_file = './Dataset/
DEBUG_MODE=False

class Expression(data.Dataset):
    def __init__(self, args, train=True):
        csv_data = pd.read_csv('./Dataset/encoded_dataset.csv')
        encoded = csv_data['encoded']
        encoded = encoded.values
        self.encoded_list = []
        for encoding in encoded:
            encoding = encoding.replace("[", "")
            encoding = encoding.replace("]", "")
            encoding = encoding.replace("\n", "")
            encoding = encoding.split()
            #print(encoding)
            encoding = np.array(encoding).astype(int)
            # Add start and end token
            encoding = np.append(1, encoding)
            encoding = np.append(encoding, 2)
            self.encoded_list.append(encoding)

        self.image_paths = csv_data['image_paths']
        self.image_paths = self.image_paths.values.tolist()
        self.args = args
        self.train = train
        self.expression_train, self.expression_test, self.label_train, self.label_test = \
            train_test_split(self.image_paths, self.encoded_list, test_size=0.1, random_state=args.seed)

    def __getitem__(self, idx):
        if not self.train:
            image = imageio.imread(os.path.join('./Dataset/', self.expression_test[idx]))
            image = common.normalize_img(image)
            image = common.exp_rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            label = torch.Tensor(self.label_test[idx])
            filename = self.expression_test[idx]
            return filename, image, label

        else:
            idx = idx % len(self.expression_train)
            image = imageio.imread('./Dataset/' + self.expression_train[idx])
            image = common.normalize_img(image)
            image = common.exp_rand_place(image)
            image = transforms.ToTensor()(image[:, :, np.newaxis])
            label = torch.Tensor(self.label_train[idx])
            if DEBUG_MODE:
                print("Label: ", self.label_train[idx])
            filename = self.expression_train[idx]

            return filename, image, label

    def __len__(self):
        if not self.train:
            # return len(self.expression_train)
            return len(self.expression_test)
        else:
            return len(self.expression_train)
            # return 16

