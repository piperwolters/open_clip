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
        self.image_path = opt['image_path']
        self.example_ids_fname = opt['example_ids']

        with open(self.example_ids_fname, 'r') as f:
            self.example_ids = json.load(f)

        self.data_len = len(self.example_ids)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def transform_image(self, image):
        image = skimage.transform.resize(image, (128, 128), preserve_range=True).astype(np.uint8)
        crop_size = random.randint(112, 128)
        i = random.randint(0, image.shape[0] - crop_size)
        j = random.randint(0, image.shape[1] - crop_size)
        crop = image[i:i+crop_size, j:j+crop_size]
        crop = skimage.transform.resize(crop, (224, 224), preserve_range=True).astype(np.uint8)
        return crop

    def __getitem__(self, index):
        example_id = self.example_ids[index]
        fnames = ['naip.png', 'good.png', 'bad.png']
        img_paths = [os.path.join(self.image_path, example_id, fname) for fname in fnames]
        ims = [skimage.io.imread(img_path) for img_path in img_paths]
        ims = [totensor(self.transform_image(im)) for im in ims]
        return ims[0], ims[1], ims[2]

    def __len__(self):
        return self.data_len
