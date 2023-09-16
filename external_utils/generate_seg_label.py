import cv2
import numpy as np
import os
from tqdm import tqdm

def make_circle_mask():
    circle_mask = np.zeros(shape=(1024, 1280, 3), dtype=np.uint8)
    circle_mask = cv2.circle(circle_mask, center=(640, 512), radius=int(360 * np.sqrt(2)),
                             color=(255, 255, 255), thickness=-1)
    circle_mask = (circle_mask > 0).astype(np.uint8)
    return circle_mask

def crop(img_path, circle_mask):

    img = cv2.imread(img_path)
    # img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_LINEAR)

    img = img * circle_mask

    return img

def read_segmentation_image(segmentation_image_path, visualize=False):
    circle_mask = make_circle_mask()

    seg_img = crop(segmentation_image_path, circle_mask)
    if visualize is True:
        cv2.imshow('img', seg_img)
        cv2.waitKey(0)
    return seg_img

def convert_segmentation_image_to_label(segmentation_image, visualization=False):
    h, w, c = segmentation_image.shape
    assert c == 3
    seg_label = np.zeros(shape=(h, w)).astype(np.uint8)
    floor_mask = np.logical_and(np.logical_and(segmentation_image[:, :, 2] >= 254, segmentation_image[:, :, 1] >= 254),
                                segmentation_image[:, :, 0] <= 1)
    seg_label[floor_mask] = 2

    body_mask = np.logical_and(np.logical_and(segmentation_image[:, :, 2] >= 254, segmentation_image[:, :, 1] <= 1),
                                segmentation_image[:, :, 0] <= 1)
    seg_label[body_mask] = 1

    if visualization:
        cv2.imshow('img', (seg_label * 125).astype(np.uint8))
        cv2.waitKey(0)
    return seg_label

def process_whole_dataset(dataset_dir, image_name, depth_name, seg_name):
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
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    img_id = os.path.splitext(img_name)[0]
                    depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                    data.append({'img': img_path, 'depth': depth_path})

    return data


if __name__ == '__main__':
    segmentation_image_path = r'Z:\EgoMocap\work\synthetic\matterport_with_seg_wo_body\Vvot9Ly1tCj\pose_17_03\seg\004.png'


    segmentation_image = read_segmentation_image(segmentation_image_path=segmentation_image_path,
                                                 visualize=True)
    seg_label = convert_segmentation_image_to_label(segmentation_image, visualization=True)



