import cv2
import numpy as np
import os

def crop(img_path):
    circle_mask = np.zeros(shape=(512, 640, 3), dtype=np.uint8)
    circle_mask = cv2.circle(circle_mask, center=(320, 256), radius=int(180 * np.sqrt(2)),
                                  color=(255, 255, 255), thickness=-1)
    circle_mask = (circle_mask > 0).astype(np.uint8)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_LINEAR)

    img = img * circle_mask

    return img

if __name__ == '__main__':
    img_dir = r'X:\Mo2Cap2Plus\work\HRNet-Semantic-Segmentation\data\office_1'
    for img_name in os.listdir(os.path.join(img_dir, 'old')):
        if img_name.endswith('png') or img_name.endswith('jpg'):
            img = crop(os.path.join(img_dir, 'old', img_name))
            cv2.imwrite(os.path.join(img_dir, img_name), img)

