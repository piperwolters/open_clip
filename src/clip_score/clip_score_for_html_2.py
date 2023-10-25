import os
import sys
import cv2
import glob
import clip
import lpips
import torch
import skimage.io
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet

from metrics import *

sys.path.append('/data/piperw/open_clip_naipnaip/open_clip/src/')
from infer_utils import sr_infer
from open_clip.model import CLIP, CLIPVisionCfg


device = torch.device('cuda')

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

# LPIPS perceptual losses / scores
loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

# Initiatialize the DinoV2 model
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)

# NAIP-NAIP-CLIP 
# SAT-CLIP model config info.
naip_info = {}
naip_vision_cfg = CLIPVisionCfg(**naip_info)

# Initialize the SAT-CLIP model.
naip_clip_model = CLIP(
            embed_dim=512,
            naip_vision_cfg=naip_vision_cfg,
        ).to(device)

# Load the pretrained SAT-CLIP checkpoint. Load weights into model.
weights_path = '/data/piperw/open_clip_naipnaip/open_clip/epoch_97.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
new_state_dict = {}
for k,v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v
naip_clip_model.load_state_dict(new_state_dict)
naip_clip_model.eval()


#### This loop is specific to the html vis - uses the folder compare/ containing numbered examples with various outputs in each ####
#### Version 2 because chaos ####
d = '/data/piperw/data/results_mturk_urban/'
for i in os.listdir(d):
    print('i:', i)

    s = d + i + '/'
    print("s:", s)

    # Filepaths for all the results 
    hr = os.path.join(str(s), 'naip.png')
    hr_x4 = os.path.join(str(s), 'naip_x4_down.png')
    hr_x8 = os.path.join(str(s), 'naip_x8_down.png')
    hr_x16 = os.path.join(str(s), 'naip_x16_down.png')
    srcnn = os.path.join(str(s), 'srcnn.png')
    high = os.path.join(str(s), 'highresnet.png')
    sr3 = os.path.join(str(s), 'sr3.png')
    sr3_cfg = os.path.join(str(s), 'sr3_cfg.png')
    gan_satlas = os.path.join(str(s), 'esrgan_satlas.png')
    gan_satlas_5k = os.path.join(str(s), 'esrgan_satlas_chkpt5k.png')
    gan_satlas_50k = os.path.join(str(s), 'esrgan_satlas_chkpt50k.png')
    gan_osm = os.path.join(str(s), 'esrgan_osm.png')
    gan_osm_5k = os.path.join(str(s), 'esrgan_osm_chkpt5k.png')
    gan_osm_50k = os.path.join(str(s), 'esrgan_osm_chkpt50k.png')

    base_paths = [hr_x4,hr_x8,hr_x16,srcnn,high,sr3,sr3_cfg,gan_satlas,gan_satlas_5k,gan_satlas_50k,
                    gan_osm,gan_osm_5k,gan_osm_50k]

    # Read in the files so metrics can be computed on the fly
    hr_im = skimage.io.imread(hr)
    hr_x4_im = skimage.io.imread(hr_x4)
    hr_x8_im = skimage.io.imread(hr_x8)
    hr_x16_im = skimage.io.imread(hr_x16)
    srcnn_im = cv2.resize(skimage.io.imread(srcnn), (128,128))
    high_im = cv2.resize(skimage.io.imread(high), (128,128))
    sr3_im = skimage.io.imread(sr3)
    sr3_cfg_im = skimage.io.imread(sr3_cfg)
    gan_satlas_im = skimage.io.imread(gan_satlas)
    gan_satlas_5k_im = skimage.io.imread(gan_satlas_5k)
    gan_satlas_50k_im = skimage.io.imread(gan_satlas_50k)
    gan_osm_im = skimage.io.imread(gan_osm)
    gan_osm_5k_im = skimage.io.imread(gan_osm_5k)
    gan_osm_50k_im = skimage.io.imread(gan_osm_50k)

    outputs = [hr_x4_im,hr_x8_im,hr_x16_im,srcnn_im,high_im,sr3_im,sr3_cfg_im,gan_satlas_im,
                gan_satlas_5k_im,gan_satlas_50k_im,gan_osm_im,gan_osm_5k_im,gan_osm_50k_im]
    print(len(outputs), " outputs to be run through series of metrics and scoring computation.")

    # Compute PSNR and SSIM for each model output
    psnrs = []
    ssims = []
    for o in outputs:
        psnr = round(calculate_psnr(hr_im, o, 0), 4)
        ssim = round(calculate_ssim(hr_im, o, 0), 4)
        psnrs.append(psnr)
        ssims.append(ssim)

    # Compute Perceptual Metric with alexnet and vgg
    alexs = []
    vggs = []
    normalized_hr = 2*(hr_im - np.amin(hr_im)) / (np.amax(hr_im) - np.amin(hr_im))-1
    hr_tensor = torch.permute(torch.tensor(normalized_hr), (2, 1, 0)).unsqueeze(0).float().to(device)
    normalized_outputs = [2*(img_array - np.amin(img_array)) / (np.amax(img_array) - np.amin(img_array))-1 for img_array in outputs]
    output_tensors = [torch.permute(torch.tensor(o), (2, 1, 0)).unsqueeze(0).float().to(device) for o in normalized_outputs]
    for o_tensor in output_tensors:
        alex = loss_fn_alex(hr_tensor, o_tensor)
        vgg = loss_fn_vgg(hr_tensor, o_tensor)
        alexs.append(alex.item())
        vggs.append(vgg.item())

    # Compute similarity scores between NAIP & model output features, using CLIP (actual CLIP)
    naip_im = preprocess(Image.open(hr)).unsqueeze(0).to(device)
    naip_feats = actual_clip_model.encode_image(naip_im)
    clip_scores = []
    for bp in base_paths:
        out_im = preprocess(Image.open(bp)).unsqueeze(0).to(device)
        print(out_im.shape, naip_im.shape)
        out_feats = actual_clip_model.encode_image(out_im)
        print(out_feats.shape, naip_feats.shape)
        print("ranges of out:", torch.min(out_im), torch.max(out_im))
        print("ranges of hr:", torch.min(naip_im), torch.max(naip_im))
        sim = str(F.cosine_similarity(naip_feats, out_feats).detach().item())
        print("sim:", sim)
        clip_scores.append(sim)

    # Compute similarity scores between NAIP and model output features, using NAIP-NAIP-CLIP
    naip_clip_scores = []
    naip_feat = naip_clip_model.encode_image(hr_tensor)
    for o_tensor in output_tensors:
        o_feat = naip_clip_model.encode_image(o_tensor)

        sim = str(F.cosine_similarity(naip_feat, o_feat).detach().item())
        print("sim:", sim)
        naip_clip_scores.append(sim)

    # Compute similarity scores between NAIP and model output features, using DinoVs
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    hr_resized = torch.nn.functional.interpolate(hr_tensor, (126,126))
    dino_scores = []
    for o_tensor in output_tensors:
        # Resize tensors to have dimensions divisible by 14.
        o_resized = torch.nn.functional.interpolate(o_tensor, (126,126))

        hr_feat = dinov2_vitg14(hr_resized)
        o_feat = dinov2_vitg14(o_resized)

        sim = str(F.cosine_similarity(hr_feat, o_feat).detach().item())
        dino_scores.append(sim)


    with open(base_path + '/cos_sims.txt', 'w') as f:
        f.write(s2_naip+', '+s2_high+', '+s2_sr3+', '+s2_gan+', '+clip_s2_high+', '+clip_s2_sr3+', '+clip_s2_gan)

