
import argparse
import sys
sys.path.append('lib')
import numpy as np
import torch
from torch.nn import functional as F
import os
import models
from config import config
from config import update_config
from datasets.egocentric import EgocentricDemo
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        default='experiments/egocentric/egocentric_seg_test.yaml',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--img_dir', default='data/example_data/imgs', type=str, required=False)
    parser.add_argument('--model_path', default='checkpoints/ego_human_seg.pth.tar', type=str, required=True)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


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

            base_out = os.path.join(os.path.split(os.path.split(path)[0])[0], save_name)
            if not os.path.isdir(base_out):
                os.mkdir(base_out)
            # confidence_out = os.path.join(base_out, 'confidence')
            # if not os.path.isdir(confidence_out):
            #     os.mkdir(confidence_out)
            # matplotlib.image.imsave(os.path.join(confidence_out, os.path.split(path)[1]), pred)

            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0

            # overlay
            img = cv2.imread(path)
            img = img[:, :, ::-1]
            pred = cv2.resize(pred, (img.shape[0], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            pad_num = (img.shape[1] - img.shape[0]) // 2
            pred = np.pad(pred, ((0, 0), (pad_num, pad_num)), 'constant', constant_values=0)

            mask = pred

            pred = np.stack([np.zeros_like(pred), pred, np.zeros_like(pred), pred], axis=2)
            plt.imshow(img)  # Also set the cmap to gray
            plt.imshow(pred, alpha=0.4)

            overlay_out = os.path.join(base_out, 'overlay')
            if not os.path.isdir(overlay_out):
                os.mkdir(overlay_out)
            plt.savefig(os.path.join(overlay_out, os.path.split(path)[1]))

            mask = np.stack([mask, mask, mask], axis=2)
            img = (img * mask).astype(np.uint8)
            seg_out = os.path.join(base_out, 'segs')
            if not os.path.isdir(seg_out):
                os.mkdir(seg_out)
            plt.imsave(os.path.join(seg_out, os.path.split(path)[1]), img)
            plt.close()

            # print("processed image {}".format(path))


def main():
    args = parse_args()

    # build model
    model = models.seg_hrnet.get_seg_model(config)
    model_state_file = args.model_path
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
        num_workers=config.WORKERS,
        pin_memory=True)

    validate(testloader, model, save_name='segs')


if __name__ == '__main__':
    main()
