
import argparse

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
    parser.add_argument('--img_dir', default=r'X:\Mo2Cap2Plus1\static00\ExternalEgo\olek_outdoor\imgs', type=str, required=False)
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


def save_seg(seg_image, out_path):
    # seg from 0-1 to rgb
    seg_image = cv2.resize(seg_image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    seg_image = np.pad(seg_image, ((0, 0), (128, 128)), 'constant', constant_values=0)
    save_img = np.zeros(shape=(seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)
    save_img[:, :, 2] = 255 * seg_image

    cv2.imwrite(out_path, save_img)


def validate(testloader, model, save_name):
    model.eval()

    with torch.no_grad():
        for _, (image, path) in tqdm(enumerate(testloader)):
            image = image.cuda()
            path = path[0]
            size = image.shape

            pred = model(image)
            pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
            pred = F.softmax(pred, dim=1)

            pred = pred.squeeze()[1].cpu().numpy()

            base_out = os.path.join(os.path.split(os.path.split(path)[0])[0], 'segs')
            if not os.path.isdir(base_out):
                os.mkdir(base_out)

            seg_out_path = os.path.join(base_out, os.path.split(path)[1])

            # confidence_out = os.path.join(base_out, 'confidence')
            # if not os.path.isdir(confidence_out):
            #     os.mkdir(confidence_out)
            # matplotlib.image.imsave(os.path.join(confidence_out, os.path.split(path)[1]), pred)

            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0

            save_seg(pred, seg_out_path)

            # # overlay

            # img = cv2.imread(path)
            # img = img[:, :, ::-1]
            # pred = cv2.resize(pred, (img.shape[0], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            # pad_num = (img.shape[1] - img.shape[0]) // 2
            # pred = np.pad(pred, ((0, 0), (pad_num, pad_num)), 'constant', constant_values=0)
            #
            # mask = pred
            #
            # pred = np.stack([np.zeros_like(pred), pred, np.zeros_like(pred), pred], axis=2)
            # plt.imshow(img)  # Also set the cmap to gray
            # plt.imshow(pred, alpha=0.4)
            #
            # overlay_out = os.path.join(base_out, 'overlay')
            # if not os.path.isdir(overlay_out):
            #     os.mkdir(overlay_out)
            # plt.savefig(os.path.join(overlay_out, os.path.split(path)[1]))
            #
            # mask = np.stack([mask, mask, mask], axis=2)
            # img = (img * mask).astype(np.uint8)
            # seg_out = os.path.join(base_out, 'segs')
            # if not os.path.isdir(seg_out):
            #     os.mkdir(seg_out)
            # plt.imsave(os.path.join(seg_out, os.path.split(path)[1]), img)
            # plt.close()

            # print("processed image {}".format(path))


def main():
    import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    test_dataset = EgocentricDemo(
        root=args.img_dir,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    validate(testloader, model, save_name=args.model_name)


if __name__ == '__main__':
    main()
