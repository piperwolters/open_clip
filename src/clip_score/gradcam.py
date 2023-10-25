import os
import sys
import cv2
import glob
import torch
import skimage.io
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import clip
from open_clip.model import CLIP, CLIPVisionCfg

device = torch.device('cuda')

# CLIP model config info.
s2_info = {}
s2_vision_cfg = CLIPVisionCfg(**s2_info)
naip_info = {}
naip_vision_cfg = CLIPVisionCfg(**naip_info)
# Initialize the CLIP model.
clip_model = CLIP(
            embed_dim=512,
            #s2_vision_cfg=s2_vision_cfg,
            naip_vision_cfg=naip_vision_cfg,
        )
# Load the pretrained CLIP checkpoint. Load weights into model.
#weights_path = '/data/piperw/open_clip/src/logs/8gpus-1s2-1024batch-2workers-1e4lr-newvalset/checkpoints/epoch_16.pt'
#weights_path = '/home/favyenb/open_clip/src/logs/4gpus-1s2-1024batch-2workers-1e4lr-newvalset-urban/checkpoints/epoch_323.pt'
#weights_path = '/home/favyenb/open_clip/src/logs/8gpus-1s2-2048batch-2workers-1e4lr-newvalset-urban/checkpoints/epoch_227.pt'
weights_path = '/home/favyenb/open_clip_naip_contrast/src/logs/naiponly-urban-8gpu-1024batch-2worker-1e4lr/checkpoints/epoch_84.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
state_dict = {k[7:]: v for k, v in state_dict.items()}
clip_model.load_state_dict(state_dict)
clip_model.eval()
clip_model.to(device)

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        return 1 - torch.nn.functional.cosine_similarity(model_output, self.features, dim=0)

class SimilarToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        return torch.nn.functional.cosine_similarity(model_output, self.features, dim=0)

class ActualClipModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

class ClipModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_naip(image)

def reshape_transform(tensor, height=7, width=7):
    result = tensor[1:, :, :].reshape(1, height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result.float()

def reshape_transform_satclip(tensor, height=8, width=8):
    result = tensor[1:, :, :].reshape(1, height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result.float()

d = 'compare/'

'''
with GradCAM(
    model=ActualClipModelWrapper(actual_clip_model),
    target_layers=[actual_clip_model.visual.transformer._modules['resblocks'][11].ln_1],
    use_cuda=True,
    reshape_transform=reshape_transform) as cam:
    for i in range(0, 30):
        base_path = d + str(i)
        if not os.path.exists(base_path):
            continue

        naip = skimage.io.imread(base_path + '/hr.png')
        high = skimage.io.imread(base_path + '/highresnet.png')
        sr3 = skimage.io.imread(base_path + '/sr3_cfg.png')
        gan = skimage.io.imread(base_path + '/esrgan.png')

        # Load, preprocess, and feed images through actual CLIP
        naip_im = preprocess(Image.fromarray(naip)).unsqueeze(0).to(device)
        high_im = preprocess(Image.fromarray(high)).unsqueeze(0).to(device)
        sr3_im = preprocess(Image.fromarray(sr3)).unsqueeze(0).to(device)
        gan_im = preprocess(Image.fromarray(gan)).unsqueeze(0).to(device)

        naip_feats = actual_clip_model.encode_image(naip_im)[0, :]
        cam_targets = [SimilarToConceptTarget(naip_feats)]

        high_cam = cam(input_tensor=high_im, targets=cam_targets)[0]
        sr3_cam = cam(input_tensor=sr3_im, targets=cam_targets)[0]
        gan_cam = cam(input_tensor=gan_im, targets=cam_targets)[0]

        high_cam_im = show_cam_on_image(skimage.transform.resize(high, (224, 224), preserve_range=True)/255, high_cam, use_rgb=True)
        sr3_cam_im = show_cam_on_image(skimage.transform.resize(sr3, (224, 224), preserve_range=True)/255, sr3_cam, use_rgb=True)
        gan_cam_im = show_cam_on_image(skimage.transform.resize(gan, (224, 224), preserve_range=True)/255, gan_cam, use_rgb=True)

        skimage.io.imsave('compare/{}/highresnet_grid.png'.format(i), high_cam_im)
        skimage.io.imsave('compare/{}/sr3_cfg_grid.png'.format(i), sr3_cam_im)
        skimage.io.imsave('compare/{}/esrgan_grid.png'.format(i), gan_cam_im)
'''

with GradCAM(
    model=ClipModelWrapper(clip_model),
    #target_layers=[clip_model.naip_visual.transformer.resblocks[11].ln_1],
    target_layers=[clip_model.visual.transformer.resblocks[11].ln_1],
    use_cuda=True,
    reshape_transform=reshape_transform_satclip) as cam:
    for i in range(0, 30):
        base_path = d + str(i)
        if not os.path.exists(base_path):
            continue

        naip = skimage.io.imread(base_path + '/hr.png')
        high = skimage.io.imread(base_path + '/highresnet.png')
        sr3 = skimage.io.imread(base_path + '/sr3_cfg.png')
        gan = skimage.io.imread(base_path + '/esrgan.png')

        # Load, preprocess, and feed images through actual CLIP
        naip_im = torch.tensor(np.transpose(naip, (2, 0, 1))).unsqueeze(0).float().to(device)/255
        high_im = torch.tensor(np.transpose(high, (2, 0, 1))).unsqueeze(0).float().to(device)/255
        sr3_im = torch.tensor(np.transpose(sr3, (2, 0, 1))).unsqueeze(0).float().to(device)/255
        gan_im = torch.tensor(np.transpose(gan, (2, 0, 1))).unsqueeze(0).float().to(device)/255

        naip_feats = clip_model.encode_naip(naip_im)[0, :]
        cam_targets = [DifferenceFromConceptTarget(naip_feats)]

        high_cam = cam(input_tensor=high_im, targets=cam_targets)[0]
        sr3_cam = cam(input_tensor=sr3_im, targets=cam_targets)[0]
        gan_cam = cam(input_tensor=gan_im, targets=cam_targets)[0]

        high_cam_im = show_cam_on_image(high.astype(np.float32)/255, high_cam, use_rgb=True)
        sr3_cam_im = show_cam_on_image(sr3.astype(np.float32)/255, sr3_cam, use_rgb=True)
        gan_cam_im = show_cam_on_image(gan.astype(np.float32)/255, gan_cam, use_rgb=True)

        skimage.io.imsave('compare/{}/highresnet_grid.png'.format(i), high_cam_im)
        skimage.io.imsave('compare/{}/sr3_cfg_grid.png'.format(i), sr3_cam_im)
        skimage.io.imsave('compare/{}/esrgan_grid.png'.format(i), gan_cam_im)
