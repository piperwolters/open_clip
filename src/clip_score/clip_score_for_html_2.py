import os
import sys
import cv2
import timm
import glob
import clip
import json
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

sys.path.append('/data/piperw/satlas-projects/satlas/')
from satlas.model.model import Model


device = torch.device('cuda')

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load pretrained SigLIP (using open_clip)
#siglip_model, siglip_preprocess = create_model_from_pretrained('hf-hub:ViT-B-16-SigLIP')
# Load pretrained SigLIP (using timm)
siglip_model = timm.create_model(
    'vit_base_patch16_siglip_224',
    pretrained=True,
    num_classes=0,
).eval().to(device)
data_config = timm.data.resolve_model_data_config(siglip_model)
siglip_transforms = timm.data.create_transform(**data_config, is_training=False)

# LPIPS perceptual losses / scores
loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

# Initiatialize the DinoV2 model
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)

# NAIP-NAIP-CLIP 
# SAT-CLIP model config info.
naip_info = {}
naip_vision_cfg = CLIPVisionCfg(**naip_info)

# Initialize the NAIP-CLIP model.
naip_clip_model = CLIP(
            embed_dim=512,
            naip_vision_cfg=naip_vision_cfg,
        ).to(device)

# Load the pretrained NAIP-CLIP checkpoint. Load weights into model.
weights_path = '/data/piperw/open_clip_naipnaip/open_clip/epoch_97.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
new_state_dict = {}
for k,v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v
naip_clip_model.load_state_dict(new_state_dict)
naip_clip_model.eval()

# Satlas model
satlas_cfg_path = '/data/piperw/satlas-projects/satlas/old_mi.txt'
with open(satlas_cfg_path, 'r') as f:
    satlas_cfg = json.load(f)
satlas_model = Model({'config': satlas_cfg['Model'], 'channels': ['tci'], 'tasks': satlas_cfg['Tasks']}).to(device)
satlas_weights_path = '/data/piperw/satlas-projects/satlas/satlas.pth'
satlas_weights = torch.load(satlas_weights_path)
satlas_model.load_state_dict(satlas_weights, strict=False)
satlas_backbone = satlas_model.backbone
satlas_intermediates = satlas_model.intermediates


#### This loop is specific to the html vis - uses the folder compare/ containing numbered examples with various outputs in each ####
#### Version 2 because chaos ####
results = {}

d = '/data/piperw/data/results_mturk_urban/'
for i in os.listdir(d):
    if i == '__pycache__':
        continue
    if not os.path.isdir(d+i):
        continue

    print('Processing.....', i)
    results[str(i)] = {}

    s = d + i + '/'

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
        psnr = round(calculate_psnr(hr_im, o, 0), 3)
        ssim = round(calculate_ssim(hr_im, o, 0), 3)
        psnrs.append(psnr)
        ssims.append(ssim)
    results[str(i)]['psnr'] = psnrs
    results[str(i)]['ssim'] = ssims

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
        alexs.append(round(alex.item(), 3))
        vggs.append(round(vgg.item(), 3))
    results[str(i)]['lpips_alex'] = alexs
    results[str(i)]['lpips_vgg'] = vggs

    # Compute similarity scores between NAIP & model output features, using CLIP (actual CLIP)
    naip_im = preprocess(Image.open(hr)).unsqueeze(0).to(device)
    naip_feats = actual_clip_model.encode_image(naip_im)
    clip_scores = []
    for bp in base_paths:
        out_im = preprocess(Image.open(bp)).unsqueeze(0).to(device)
        out_feats = actual_clip_model.encode_image(out_im)
        sim = str(round(F.cosine_similarity(naip_feats, out_feats).detach().item(), 3))
        clip_scores.append(sim)
    results[str(i)]['clip'] = clip_scores

    # Compute similarity scores between NAIP and model output features, using NAIP-NAIP-CLIP
    naip_clip_scores = []
    naip_feat = naip_clip_model.encode_image(hr_tensor)
    for o_tensor in output_tensors:
        o_feat = naip_clip_model.encode_image(o_tensor)
        sim = str(round(F.cosine_similarity(naip_feat, o_feat).detach().item(), 3))
        naip_clip_scores.append(sim)
    results[str(i)]['naip_clip'] = naip_clip_scores

    # Compute similarity scores between NAIP and model output features, using SigLIP
    siglip_scores = []
    pil_hr = Image.open(hr)
    naip_siglip = siglip_transforms(pil_hr).unsqueeze(0).to(device)
    naip_feat = siglip_model(naip_siglip)
    for bp in base_paths:
        pil_o = Image.open(bp)
        o_siglip = siglip_transforms(pil_o).unsqueeze(0).to(device)
        o_feat = siglip_model(o_siglip)
        sim = str(round(F.cosine_similarity(naip_feat, o_feat).detach().item(), 3))
        siglip_scores.append(sim)
    results[str(i)]['siglip'] = siglip_scores

    # Compute similarity scores between NAIP and model output features, using DinoVs
    hr_resized = torch.nn.functional.interpolate(hr_tensor, (126,126))
    dino_scores = []
    for o_tensor in output_tensors:
        # Resize tensors to have dimensions divisible by 14.
        o_resized = torch.nn.functional.interpolate(o_tensor, (126,126))

        hr_feat = dinov2_vitg14(hr_resized)
        o_feat = dinov2_vitg14(o_resized)

        sim = str(round(F.cosine_similarity(hr_feat, o_feat).detach().item(), 3))
        dino_scores.append(sim)
    results[str(i)]['dino'] = dino_scores

    # Compute similarity scores between NAIP and model output features, using Satlas
    hr_tensor = torch.permute(torch.tensor(hr_im), (2, 0, 1)).unsqueeze(0).float().to(device)
    hr_backbone = satlas_backbone(hr_tensor)
    hr_intermed = satlas_intermediates(hr_backbone)
    satlas_backbone0_scores, satlas_backbone3_scores = [], []
    satlas_intermed0_scores, satlas_intermed8_scores = [], []
    for o in outputs:
        o_tensor = torch.permute(torch.tensor(o), (2, 0, 1)).unsqueeze(0).float().to(device)
        o_backbone = satlas_backbone(o_tensor)
        o_intermed = satlas_intermediates(o_backbone)

        sim_backbone0 = str(round(torch.mean(F.cosine_similarity(hr_backbone[0], o_backbone[0])).detach().item(), 3))
        sim_backbone3 = str(round(torch.mean(F.cosine_similarity(hr_backbone[3], o_backbone[3])).detach().item(), 3))
        sim_intermed0 = str(round(torch.mean(F.cosine_similarity(hr_intermed[0], o_intermed[0])).detach().item(), 3))
        sim_intermed8 = str(round(torch.mean(F.cosine_similarity(hr_intermed[8], o_intermed[8])).detach().item(), 3))

        satlas_backbone0_scores.append(sim_backbone0)
        satlas_backbone3_scores.append(sim_backbone3)
        satlas_intermed0_scores.append(sim_intermed0)
        satlas_intermed8_scores.append(sim_intermed8)

    results[str(i)]['satlas_bck0'] = satlas_backbone0_scores
    results[str(i)]['satlas_bck3'] = satlas_backbone3_scores
    results[str(i)]['satlas_int0'] = satlas_intermed0_scores
    results[str(i)]['satlas_int8'] = satlas_intermed8_scores


with open('results.json', 'w') as json_file:
    json.dump(results, json_file)
