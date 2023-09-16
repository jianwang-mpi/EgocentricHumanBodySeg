import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .base_dataset import BaseDataset
import json
import pickle

class EgocentricWeipeng(BaseDataset):
    def __init__(self,
                 root,
                 list_path='matterport_train.npy',
                 num_samples=None,
                 num_classes=2,
                 multi_scale=False,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(EgocentricWeipeng, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        # read image hdf5 file list
        with open(os.path.join(self.root, 'filenames.json')) as f:
            self.img_list = json.load(f)
        if 'test' in self.list_path:
            print('load test data')
            self.img_list = self.img_list[-1000:]
        else:
            print('load training data')
            self.img_list = self.img_list[:-1000]

    # def read_files(self):
    #     print('reading images!')
    #
    #     for i, hdf5_name in tqdm(enumerate(self.img_list)):
    #         hdf5_path = os.path.join(self.root, hdf5_name)
    #         data = h5py.File(hdf5_path, mode='r')
    #         images = data['image']                      # get input images (bgr)
    #         labels = data['foreground'][:, 3, :, :]       # get segmentation label (0-255)
    #
    #         images = np.transpose(images, (0, 2, 3, 1))
    #
    #         # crop center for image and label
    #         images = images[:, :, 32: -32, :]
    #         labels = labels[:, :, 32: -32]
    #
    #         # change the label to 0-1
    #         labels = (labels / 255).astype(np.uint8)
    #
    #         self.images[i * 1000: (i + 1) * 1000] = images
    #         self.labels[i * 1000: (i + 1) * 1000] = labels
    #
    #         del data
    #
    #     print("read files complete!")

    def __len__(self):
        return len(self.img_list)

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        name = str(index)

        training_file_name = self.img_list[index]
        training_file_path = os.path.join(self.root, training_file_name)
        with open(training_file_path, 'rb') as f:
            d = pickle.load(f)

        image = d['Image']
        label = d['Foreground'][3, :, :]

        image = np.transpose(image, (1, 2, 0))

        # crop center for image and label
        image = image[:, 32: -32, :]
        label = label[:, 32: -32]

        # change the label to 0-1
        label = (label / 255).astype(np.uint8)

        size = label.shape

        if 'test' in self.list_path:
            image = cv2.resize(image, self.crop_size,
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name
