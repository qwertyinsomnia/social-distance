import os
import sys
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict
from typing import List

import h5py
import numpy as np
import pandas
import pickle

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import hflip

import cv2

class CityUHKBEV(Dataset):
    normalize = Normalize(mean=[0.49124524, 0.47874022, 0.4298056],
                          std=[0.21531576, 0.21034797, 0.20407718])

    available_keys = [
        'bev_center', 'bev_coord', 'bev_map', 'bev_scale',
        'camera_angle', 'camera_fu', 'camera_fv', 'camera_height',
        'feet_annotation', 'feet_map',
        'head_annotation', 'head_map',
        'image',
        'num_annotations',
        'roi_mask',
        'world_coord',
    ]

    def __init__(self, root, datalist, keys: list, use_augment: bool = True):
        super(CityUHKBEV, self).__init__()

        self.root = root
        self.datalist = datalist

        self.keys = list(set(keys))
        self.load_image = 'image' in self.keys

        self.use_augment = use_augment
        self.do_normalization = True

    @staticmethod
    def to_tensor(x, dtype=None):
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype)

    def augment(self, out):
        if not self.use_augment or torch.rand(1).item() < 0.5:
            return out
        else:
            # random horizontal flip
            for key in ['image', 'feet_map', 'head_map', 'bev_map']:
                if key in out:
                    out[key] = hflip(out[key])
            for key in ['feet_annotation', 'head_annotation']:
                if key in out:
                    n = out['num_annotations'].int()
                    out[key][0, :n] = out['image'].size(-1) - out[key][0, :n]
            return out

    def __getitem__(self, item):
        scene_id = self.datalist[item][0]
        image_id = self.datalist[item][1]

        scene_name = f"scene_{scene_id:03d}"

        scene_file = os.path.join(self.root, f"{scene_name}.h5")
        scene_data = h5py.File(scene_file, "r")

        out = dict(
            image_id=self.to_tensor(image_id),
            scene_id=self.to_tensor(scene_id)
        )

        for key in self.keys:
            out[key] = self.to_tensor(scene_data[key][image_id])

        if self.load_image:
            out['image'] = self.to_tensor(scene_data['image'][image_id]) / 255.
            out = self.augment(out)
            if self.do_normalization:
                out['image'] = self.normalize(out['image'])

        scene_data.close()

        return out

    def __len__(self):
        # if is_debug():
        #     return get_debug_size()
        return len(self.datalist)

def load_path_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_datalist(root, mixed=False):
    datalist_file = "scene-mixed" if mixed else "scene-split"
    datalist_file = f"{datalist_file}.datalist"
    datalist_file = os.path.join(root, datalist_file)
    datalist = load_path_data(datalist_file)
    return datalist

if __name__=='__main__':
    # input_h5_file_path = "dataset/CityUHK-X-BEV-master/CityUHK-X-BEV/scene_002.h5"
    # input_h5 = h5py.File(input_h5_file_path, 'r')


    # input_images_data = input_h5['image']
    # in_height, in_width = input_images_data.shape[2:]
    # print(input_images_data.shape)

    # for i in range(5):
    #     image = np.array(input_images_data[i])
    #     image = image.transpose(1, 2, 0)
    #     print(image.shape)

    #     cv2.imshow("win", image)
    #     cv2.waitKey(0)


    # camera_angles = input_h5['camera_angle']
    # print(list(camera_angles))
    # camera_heights = input_h5['camera_height']
    # print(list(camera_heights))
    # camera_fus = input_h5['camera_fu']
    # print(list(camera_fus))
    # camera_fvs = input_h5['camera_fv']
    # print(list(camera_fvs))

    # input_h5.close()
    root = "dataset/CityUHK-X-BEV-master/CityUHK-X-BEV"
    train_key = 'train'
    valid_ratio = 0.2
    keys = ['head_map', 'feet_map', 'bev_map', 'camera_height', 'camera_angle', 'camera_fu', 'camera_fv']
    use_augment = True

    datalist = load_datalist(root, True)
    # print(datalist[train_key])
    num_train = len(datalist[train_key]) * (1 - valid_ratio)
    num_train = int(num_train)
    train_datalist = datalist[train_key][:num_train]

    train_dataset = CityUHKBEV(root, train_datalist, keys, use_augment)
    print(len(train_dataset))
