import random
import torch
from PIL import Image
from glob import glob
import os
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

MASK_EXTENSIONS = [
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_mask_file(filename):
    return any(filename.endswith(extension) for extension in MASK_EXTENSIONS)


class trainset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform):
        super(trainset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.paths = []
        for root, _, fnames in os.walk(os.path.dirname(os.path.abspath(__file__)) + '/' + img_root):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.paths.append(path)

        self.mask_paths = []
        for root, _, fnames in os.walk(os.path.dirname(os.path.abspath(__file__)) + '/' + mask_root):
            for fname in fnames:
                if is_mask_file(fname):
                    path = os.path.join(root, fname)
                    self.mask_paths.append(path)
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)


class testset(torch.utils.data.Dataset):
    def __init__(self, list_file, img_transform, mask_transform,
                 return_name=False):
        super(testset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.return_name = return_name

        self.datas = []
        with open(list_file) as f:
            for line in f:
                self.datas.append(line.strip())

    def __getitem__(self, index):
        data = self.datas[index]
        gt_name = data.split('\t')[0]
        mask_name = data.split('\t')[1]
        gt_img = Image.open(gt_name)
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(mask_name)
        mask = self.mask_transform(mask.convert('RGB'))
        if self.return_name:
            return gt_img * mask, mask, gt_img, gt_name
        else:
            return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.datas)
