import sys
sys.path.append('../lib')
import argparse
import pickle

import matplotlib
import matplotlib.image
import numpy as np
import torch
from torch.nn import functional as F
import os
import _init_paths
import models
from config import config
from config import update_config
from datasets.egocentric import EgocentricDemo
import skimage.io as skio
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--data_dir', default=r'X:\Mo2Cap2Plus1\static00\EgocentricData\REC08102020\jian3', type=str, required=False)
    # parser.add_argument('--model_name', default='egocentric_adadelta_epoch20_new', type=str, required=False)
    parser.add_argument('--model_name', default='egocentric_adadelta_epoch20_weipeng', type=str, required=False)
    parser.add_argument('--model_iter', default=1000, type=int, required=False)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def validate(testloader, model, out_dir):
    model.eval()

    with torch.no_grad():
        for _, (image_batch, path_batch) in tqdm(enumerate(testloader)):
            image_batch = image_batch.cuda()
            size = image_batch.shape


            pred_batch = model(image_batch)
            pred_batch = F.upsample(input=pred_batch, size=(size[-2], size[-1]), mode='bilinear')
            pred_batch = F.softmax(pred_batch, dim=1)

            for i in range(len(pred_batch)):

                path = path_batch[i]

                pred = pred_batch[i][1].cpu().numpy()


                output_name = os.path.splitext(os.path.split(path)[1])[0]
                out_path = os.path.join(out_dir, "{}.pkl".format(output_name))
                with open(out_path, 'wb') as f:
                    pickle.dump(pred, f)



def main():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parse_args()

    # build model
    model = models.seg_hrnet.get_seg_model(config)
    # model_state_file = r'../output/egocentric/{}/checkpoint_epoch_{}.pth.tar'.format(args.model_name, args.model_iter)

    model_state_file = '../output/egocentric_weipeng/{}/checkpoint_epoch_{}.pth.tar'.format(args.model_name, args.model_iter)
    print('=> loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file)['state_dict']
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)
    model = model.cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    # run the whole data sequence
    external_data_all_path = r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_all\jianwang'
    root_dir_names = os.listdir(external_data_all_path)

    for root_dir_name in root_dir_names:
        root_dir_path = os.path.join(external_data_all_path,
                                     root_dir_name)
        print("running dir: {}".format(root_dir_path))
        if not os.path.isdir(root_dir_path):
            continue
        if "out" not in root_dir_path:
            print("do not deal with jianwang")
            continue
        # for seq_name in os.listdir(root_dir_path):
        #     seq_path = os.path.join(root_dir_path, seq_name)
        #
        #     if not os.path.isdir(seq_path):
        #         print("have already generated the segs")
        #         continue

        print("running seq: {}".format(root_dir_path))

        image_dir = os.path.join(root_dir_path, 'imgs')
        out_dir = os.path.join(root_dir_path, 'segs')
        if os.path.isdir(out_dir):
            continue
        os.mkdir(out_dir)

        test_dataset = EgocentricDemo(
            root=image_dir,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TEST.BASE_SIZE,
            crop_size=test_size,
            downsample_rate=1)

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)

        validate(testloader, model, out_dir)


if __name__ == '__main__':
    main()
