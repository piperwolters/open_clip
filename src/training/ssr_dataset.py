import cv2
import glob
import json
from multiprocessing import Manager
import numpy as np
import os
from PIL import Image
import random
import skimage.io
import skimage.transform
import torch
import torchvision
from torch.utils import data as data
from torchvision.transforms import functional as trans_fn

totensor = torchvision.transforms.ToTensor()

class SSRDataset(data.Dataset):
    def __init__(self, opt):
        super(SSRDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.n_s2_images = 1

        # Paths to Sentinel-2 and NAIP imagery.
        self.naip_path = opt['naip_path']
        self.image_list_fname = opt['image_list']
        if not os.path.exists(self.naip_path):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.tiles = {}
        with open(self.image_list_fname, 'r') as f:
            for col, row, image_id in json.load(f):
                tile = (col, row)
                if tile not in self.tiles:
                    self.tiles[tile] = []
                self.tiles[tile].append(image_id)

        for tile in list(self.tiles.keys()):
            if len(self.tiles[tile]) < 2:
                del self.tiles[tile]

        self.datapoints = list(self.tiles.keys())

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def transform_image(self, image):
        crop_size = random.randint(112, 128)
        i = random.randint(0, image.shape[0] - crop_size)
        j = random.randint(0, image.shape[1] - crop_size)
        crop = image[i:i+crop_size, j:j+crop_size]
        crop = skimage.transform.resize(crop, (128, 128), preserve_range=True).astype(np.uint8)
        return crop

    def __getitem__(self, index):
        tile = self.datapoints[index]
        image_ids = random.sample(self.tiles[tile], 2)
        random.shuffle(image_ids)
        img_path1 = os.path.join(self.naip_path, image_ids[0], 'tci', '{}_{}.png'.format(tile[0], tile[1]))
        img_path2 = os.path.join(self.naip_path, image_ids[1], 'tci', '{}_{}.png'.format(tile[0], tile[1]))

        naip1 = skimage.io.imread(img_path1)
        naip2 = skimage.io.imread(img_path2)

        images = self.transform_image(np.concatenate([naip1, naip2], axis=2))
        img1 = totensor(images[:, :, 0:3])
        img2 = totensor(images[:, :, 3:6])
        return img1, img2

    def __len__(self):
        return self.data_len
