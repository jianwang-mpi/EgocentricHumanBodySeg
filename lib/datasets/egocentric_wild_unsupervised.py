import json
import os
import pickle
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

# import external_utils.data_transforms as transforms
from external_utils.data_transforms import Normalize, ToTensor


# from config import args, consts

class EgocentricWildDataset(Dataset):
    """
    Unsupervised dataset to read images in the wild
    """

    def __init__(self, root_data_path, image_size=(520, 520), color_jitter=False,
                 local_machine=False):

        super(EgocentricWildDataset, self).__init__()

        self.root_data_path = root_data_path
        self.is_color_jitter = color_jitter
        self.local_machine = local_machine
        self.image_size = image_size
        # get data
        self.data = []
        # identity_name_list = ['ayush', 'lingjie', 'chao', 'kripa', 'soshi']
        identity_name_list = ['ayush', 'ayush_new', 'binchen', 'chao', 'chao_new',
                              'kripa', 'kripa_new', 'lingjie', 'lingjie_new',
                              'mengyu_new', 'mohamed', 'soshi_new', 'zhili_new']
        # identity_name_list = ['ayush', 'ayush_new', 'binchen', 'chao', 'chao_new',
        #                       'kripa', 'kripa_new', 'lingjie', 'lingjie_new'
        #                       , 'mohamed', 'soshi_new']
        for identity_name in identity_name_list:
            identity_path = os.path.join(self.root_data_path, identity_name)
            self.data.extend(self.get_real_identity_data(identity_path))

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.color_jitter = ColorJitter(color_add=args.color_add, color_mul=args.color_mul)
        self.to_tensor = ToTensor()

    def get_real_data_single_seq(self, seq_dir):
        pkl_path = os.path.join(seq_dir, 'pseudo_gt.pkl')
        with open(pkl_path, 'rb') as f:
            seq_data = pickle.load(f)
        return seq_data

    def get_real_identity_data(self, identity_path):
        identity_data = []
        for seq_name in os.listdir(identity_path):
            seq_dir = os.path.join(identity_path, seq_name)
            if 'kripa' in seq_dir and 'rountunda1' in seq_dir:
                continue
            if os.path.isdir(seq_dir):
                seq_data = self.get_real_data_single_seq(seq_dir)
                identity_data.extend(seq_data)

        return identity_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]

        image_path = data_i['image_path']
        if self.local_machine:
            image_path = image_path.replace('/HPS', 'X:')
        else:
            image_path = image_path.replace('X:', '/HPS')

        img = cv2.imread(image_path)
        img = img[:, 128: -128, :]
        # data augmentation
        img = cv2.resize(img, dsize=self.image_size) / 255.
        img = self.normalize(img)
        img_torch = self.to_tensor(img)

        return img_torch


if __name__ == '__main__':
    dataset = EgocentricWildDataset(root_data_path=r'X:\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_new')
    print(len(dataset))
    # 3150
    img = dataset[68050]

    print(img.shape)
