import cv2
import os
import numpy as np
from tqdm import tqdm


def get_data(dataset_dir, image_name, depth_name):
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


def filter_image(data_dir, data_save, data_split, image_name='img', depth_name='depth'):
    data = get_data(data_dir, image_name, depth_name)
    filtered_data = []
    for item in tqdm(data):
        img_path = item['img']
        depth_path = item['depth']
        if hole_percentage(img_path, depth_path) < 0.5:
            filtered_data.append(item)
        else:
            # print('The percentage of hole of img: {} and depth: {} is over 0.5!'.format(img_path, depth_path))
            pass
    np.random.shuffle(filtered_data)
    train_data = filtered_data[:int(data_split * len(filtered_data))]
    test_data = filtered_data[int(data_split * len(filtered_data)):]
    print('saving data to: {}'.format(data_save))
    np.save(os.path.join(data_save, 'matterport_nobody_train.npy'), train_data)
    np.save(os.path.join(data_save, 'matterport_nobody_test.npy'), test_data)


def hole_percentage(img_path, depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return 1
    depth = depth[:, :, 0]

    hole = depth > 999.
    hole = hole.astype(np.float64)
    percentage = np.sum(hole) / (depth.shape[0] * depth.shape[1])
    return percentage

def calculate_floor_percentage(seg_path):
    segmentation_image = cv2.imread(seg_path)
    h, w, c = segmentation_image.shape
    assert c == 3
    seg_label = np.zeros(shape=(h, w)).astype(np.uint8)
    floor_mask = np.logical_and(
        np.logical_and(segmentation_image[:, :, 2] >= 254, segmentation_image[:, :, 1] >= 254),
        segmentation_image[:, :, 0] <= 1)
    seg_label[floor_mask] = 1

    floor_percent = np.sum(seg_label) / (h * w)

    return floor_percent




if __name__ == '__main__':
    # filter_image(r'/home/jianwang/EgoMocap/work/synthetic/depth_matterport_single_no_body',
    #              r'/home/jianwang/EgoMocap/work/synthetic/depth_matterport_single_no_body', data_split=0.9,
    #              depth_name='depth_nobody')

    # depth_path = r'Z:\EgoMocap\work\synthetic\depth_matterport_single_no_body\pa4otMbVnkk\pose_40_10\depth\012\Image0001.exr'
    # percentage = hole_percentage(None, depth_path)
    # print(percentage)

