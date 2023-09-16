import pickle

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
import cv2

image_path = r'X:\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_all\mengyu_new\rest1\imgs\img_001200.jpg'
seg_path = r'X:\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_all\mengyu_new\rest1\segs\img_001200.pkl'

with open(seg_path,'rb') as f:
    pred = pickle.load(f)

img = cv2.imread(image_path)
img = img[:, :, ::-1]


pred = cv2.resize(pred, (img.shape[0], img.shape[0]), interpolation=cv2.INTER_LINEAR)

pred[pred > 0.5] = 1
pred[pred < 0.5] = 0

pad_num = (img.shape[1] - img.shape[0]) // 2
pred = np.pad(pred, ((0, 0), (pad_num, pad_num)), 'constant', constant_values=0)

mask = pred

pred = np.stack([np.zeros_like(pred), pred, np.zeros_like(pred), pred], axis=2)
plt.imshow(img)  # Also set the cmap to gray
plt.imshow(pred, alpha=0.4)
plt.show()
