import os
import cv2
import glob
import torch
import random
import torchvision
import skimage.io
import numpy as np
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import functional as trans_fn
from basicsr.utils.registry import DATASET_REGISTRY

from multiprocessing import Manager

totensor = torchvision.transforms.ToTensor()


@DATASET_REGISTRY.register()
class SSRDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
    """

    def __init__(self, opt):
        super(SSRDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.n_s2_images = 1

        self.s2_bands = opt['s2_bands'] if 's2_bands' in opt else ['tci']

        # Paths to Sentinel-2 and NAIP imagery.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)

        if self.split == 'train':
            self.naip_chips = random.sample(self.naip_chips, 44000)

        datapoints = []
        for n in self.naip_chips:

            # Extract the X,Y tile from this NAIP image filepath.
            split_path = n.split('/')
            chip = split_path[-1][:-4]
            tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16

            # Now compute the corresponding Sentinel-2 tiles.
            s2_left_corner = tile[0] * 16, tile[1] * 16
            diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

            s2_path = [os.path.join(self.s2_path, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')]
            if not os.path.exists(s2_path[0]):
                continue

            datapoints.append([n, s2_path])

        manager = Manager()
        self.datapoints = manager.list(datapoints)

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def __getitem__(self, index):

        datapoint = self.datapoints[index]

        naip_path, s2_path = datapoint[0], datapoint[1]

        # Load the 512x512 NAIP chip.
        naip_chip = skimage.io.imread(naip_path)

        # Load the T*32x32 S2 file.
        s2_images = skimage.io.imread(s2_path[0])

        # Reshape to be Tx32x32.
        s2_chunks = np.reshape(s2_images, (-1, 32, 32, 3))

        # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
        # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
        goods, bads = [], []
        for i,ts in enumerate(s2_chunks):
            if [0, 0, 0] in ts:
                bads.append(i)
            else:
                goods.append(i)

        # Pick 18 random indices of s2 images to use. Skip ones that are partially black.
        if len(goods) >= self.n_s2_images:
            rand_indices = random.sample(goods, self.n_s2_images)
        else:
            need = self.n_s2_images - len(goods)
            rand_indices = goods + random.sample(bads, need)

        s2_chunks = [s2_chunks[i] for i in rand_indices]
        s2_chunks = np.array(s2_chunks)
        s2_chunks = [totensor(img) for img in s2_chunks]

        img_S2 = torch.cat(s2_chunks)
        img_HR = totensor(naip_chip).type(torch.FloatTensor)

        return img_S2, img_HR

    def __len__(self):
        return self.data_len
