# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from natsort import natsorted

from .base_dataset import BaseDataset


class Egocentric(BaseDataset):
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

        super(Egocentric, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        # self.img_list = [line.strip().split() for line in open(root + list_path)]
        # read image list
        self.img_list = np.load(os.path.join(root, list_path), allow_pickle=True)

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            img_path = item['img']
            label_path = item['seg']
            # change the seg to seg_label
            label_path = label_path.replace('/seg/', '/seg_label/')
            name = os.path.splitext(img_path)[0]
            sample = {"img": img_path,
                      "label": label_path,
                      "name": name}
            files.append(sample)
        print("read files complete!")
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(item["label"]), cv2.IMREAD_GRAYSCALE)
        # crop center for image and lable
        image = image[:, 64: -64, :]
        label = label[:, 64: -64]
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
            # swap left and right arm and leg
            # if flip == -1:
            #     right_idx = [15, 17, 19]
            #     left_idx = [14, 16, 18]
            #     for i in range(0, 3):
            #         right_pos = np.where(label == right_idx[i])
            #         left_pos = np.where(label == left_idx[i])
            #         label[right_pos[0], right_pos[1]] = left_idx[i]
            #         label[left_pos[0], left_pos[1]] = right_idx[i]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip=False):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred,
                          size=(size[-2], size[-1]),
                          mode='bilinear')
        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output,
                                     size=(size[-2], size[-1]),
                                     mode='bilinear')
            flip_output = flip_output.cpu().numpy()
            flip_pred = flip_output.copy()
            # flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
            # flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
            # flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
            # flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
            # flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
            # flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()


class EgocentricDemo(BaseDataset):
    def __init__(self,
                 root,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(EgocentricDemo, self).__init__(ignore_label, base_size,
                                             crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root

        self.files = self.read_files()

    def read_files(self):
        files = []
        for img_name in natsorted(os.listdir(self.root)):
            if img_name.endswith('png') or img_name.endswith('jpg'):
                img_path = os.path.join(self.root, img_name)
                files.append(img_path)
        print("read files complete!")
        return files


    def __getitem__(self, index):
        img_path = self.files[index]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(640, 512), interpolation=cv2.INTER_LINEAR)
        # crop center for image and lable
        image = image[:, 64: -64, :]

        image = cv2.resize(image, self.crop_size,
                           interpolation=cv2.INTER_LINEAR)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        return image.copy(), img_path


