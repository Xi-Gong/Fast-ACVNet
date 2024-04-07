from cgi import test
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


class KITTI3DDataset(Dataset):
    def __init__(self, kitti3d_datapath, split_file, training):
        self.datapath = kitti3d_datapath
        self.training = training
        data_prefix = self.datapath + "training/"
        self.left_filepath, self.right_filepath, self.disp_filepath = \
                self.load_path(data_prefix, split_file)
        assert self.disp_filepath is not None
    
    def load_path(self, filepath, split_file):
        left_fold = 'image_2/'
        right_fold = 'image_3/'
        disp_L = 'disparity/'

        with open(split_file, 'r') as f:
            idx = [x.strip() for x in f.readlines()]

        if not self.training:
            sorted(idx)

        left = [filepath + '/' + left_fold + img + '.png' for img in idx]
        right = [filepath + '/' + right_fold + img + '.png' for img in idx]
        disp = [filepath + '/' + disp_L + img + '.npy' for img in idx]

        return left, right, disp

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        return np.load(filename).astype(np.float32)

    def __len__(self):
        return len(self.left_filepath)

    def __getitem__(self, index):
        left_img = self.load_image(self.left_filepath[index])
        right_img = self.load_image(self.right_filepath[index])

        if self.disp_filepath:  # has disparity ground truth
            disparity = self.load_disp(self.disp_filepath[index])
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            # y1 = random.randint(0, h - crop_h)
            if  random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            disparity_low = cv2.resize(disparity, (crop_w//4, crop_h//4), interpolation=cv2.INTER_NEAREST)

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low": disparity_low}

        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filepath[index],
                        "right_filename": self.right_filepath[index]}
