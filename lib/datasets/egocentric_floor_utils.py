import cv2
import numpy as np
import os
from tqdm import tqdm


class EgocentricSegmentationPreprocess:
    def __init__(self, img_h, img_w):
        self.circle_mask = self.make_circle_mask(img_h, img_w)

    def make_circle_mask(self, img_h=1024, img_w=1280):
        circle_mask = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(img_w // 2, img_h // 2),
                                 radius=int(360 / 1024 * img_h * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.uint8)
        return circle_mask

    def crop(self, img):

        img = img * self.circle_mask

        return img

    def convert_segmentation_image_to_label(self, segmentation_image, mask_type='floor', visualization=False):
        segmentation_image = self.crop(segmentation_image)
        h, w, c = segmentation_image.shape
        assert c == 3
        seg_label = np.zeros(shape=(h, w)).astype(np.uint8)
        if mask_type == 'floor':
            mask = np.logical_and(
                np.logical_and(segmentation_image[:, :, 2] >= 254, segmentation_image[:, :, 1] >= 254),
                segmentation_image[:, :, 0] <= 1)
        elif mask_type == 'body':
            mask = np.logical_and(np.logical_and(segmentation_image[:, :, 2] >= 254, segmentation_image[:, :, 1] <= 1),
                                  segmentation_image[:, :, 0] <= 1)
        else:
            raise Exception("incorrect mask type")
        seg_label[mask] = 1

        if visualization:
            cv2.imshow('img', (seg_label * 255).astype(np.uint8))
            cv2.waitKey(0)
        return seg_label
