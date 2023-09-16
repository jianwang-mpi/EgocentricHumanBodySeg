# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F
from natsort import natsorted

from .base_dataset import BaseDataset
from .egocentric_floor_utils import EgocentricSegmentationPreprocess


class EgocentricFloor(BaseDataset):
    def __init__(self,
                 root,
                 list_path='matterport_train.npy',
                 num_samples=None,
                 num_classes=2,
                 multi_scale=False,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(520, 520),
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(EgocentricFloor, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        # self.img_list = [line.strip().split() for line in open(root + list_path)]
        # read image list
        # self.img_list = np.load(os.path.join(root, list_path), allow_pickle=True)

        self.files = self.get_image_name_list(root, image_name='img', depth_name='depth', seg_name='seg')

        self.egocentric_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)

        # self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def get_image_name_list(self, dataset_dir, image_name, depth_name, seg_name):
        data = []
        print('getting data: ')
        for scene_id in tqdm(os.listdir(dataset_dir)):
            scene_path = os.path.join(dataset_dir, scene_id)
            if os.path.isdir(scene_path) is False:
                continue
            for pose_id in os.listdir(scene_path):
                pose_path = os.path.join(scene_path, pose_id)
                if os.path.exists(os.path.join(pose_path, 'metadata.npy')):
                    img_dir = os.path.join(pose_path, image_name)
                    depth_dir = os.path.join(pose_path, depth_name)
                    seg_dir = os.path.join(pose_path, seg_name)
                    for img_name in os.listdir(img_dir):
                        img_path = os.path.join(img_dir, img_name)
                        name = os.path.splitext(img_path)[0]
                        seg_path = os.path.join(seg_dir, img_name)
                        img_id = os.path.splitext(img_name)[0]
                        depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                        data.append({'img': img_path, 'depth': depth_path, 'seg': seg_path, "name": name})

        return data


    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(item["img"]), cv2.IMREAD_COLOR)
        image = self.egocentric_process.crop(image)
        seg_image = cv2.imread(os.path.join(item["seg"]), cv2.IMREAD_COLOR)
        label = self.egocentric_process.convert_segmentation_image_to_label(seg_image, mask_type='floor')
        # crop center for image and lable
        assert image.shape[0] == 1024
        assert image.shape[1] == 1280
        assert label.shape[0] == 1024
        assert label.shape[1] == 1280
        image = image[:, 128: -128, :]
        label = label[:, 128: -128]
        size = label.shape

        if 'test' in self.list_path:
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label, self.multi_scale, False)

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
                 base_size=520,
                 crop_size=(520, 520),
                 downsample_rate=1,
                 scale_factor=11,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(EgocentricDemo, self).__init__(ignore_label, base_size,
                                             crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root

        self.files = self.read_files()

        self.egocentric_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)

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
        if image.shape[0] != 1024 or image.shape[1] != 1280:
            image = cv2.resize(image, dsize=(1280, 1024), interpolation=cv2.INTER_LINEAR)
        image = self.egocentric_process.crop(image)
        # image = cv2.resize(image, dsize=(640, 512), interpolation=cv2.INTER_LINEAR)
        # crop center for image and lable
        image = image[:, 128: -128, :]

        image = cv2.resize(image, self.crop_size,
                           interpolation=cv2.INTER_LINEAR)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        return image.copy(), img_path


