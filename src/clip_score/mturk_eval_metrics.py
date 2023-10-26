import os
import sys
import cv2
import json
import clip
import timm
import glob
import torch
import lpips
import skimage.io
import numpy as np
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from statistics import mean
from PIL import Image

from metrics import *

sys.path.append('/data/piperw/open_clip_naipnaip/open_clip/src/')
from infer_utils import sr_infer
from open_clip.model import CLIP, CLIPVisionCfg

sys.path.append('/data/piperw/satlas-projects/satlas/')
from satlas.model.model import Model


device = torch.device('cuda')

# ex: {"36005_51983": {"model1": "esrgan_osm_chkpt50k", "model2": "srcnn", "answers": [0, 0, 0, 0, 0]}
annots_file = open('/data/piperw/scripts/mturk_batch01_dict.json')
annots = json.load(annots_file)

data_dir = '/data/piperw/data/mturk_urban_outputs/'

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/16", device=device)

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

metric_names = ['psnr', 'ssim', 'lpips_alex', 'lpips_vgg', 'clip', 'naip_clip', 'siglip', 'dino', 'satlas_backbone', 'satlas_fpn']
model_names = ['srcnn','highresnet','sr3','sr3_cfg','esrgan_satlas','esrgan_satlas_chkpt5k',
                'esrgan_satlas_chkpt50k','esrgan_osm','esrgan_osm_chkpt5k','esrgan_osm_chkpt50k']

correct = {mn: 0 for mn in metric_names}
print(correct)
print("Iterating through ", len(annots.items()), " datapoints.")
for idx,(chip, d) in enumerate(annots.items()):
    print("Processing...", idx)

    human_answers = d['answers']
    avg_human = mean(human_answers)

    model1_name = d['model1']
    model2_name = d['model2']

    naip_fp = data_dir + chip + '/' + 'naip.png'
    model1_fp = data_dir + chip + '/' + model1_name + '.png'
    model2_fp = data_dir + chip + '/' + model2_name + '.png'

    naip_im = skimage.io.imread(naip_fp)
    model1_im = skimage.io.imread(model1_fp)
    model2_im = skimage.io.imread(model2_fp)

    # Account for SRCNN and HighResNet outputs being (640,640) instead of (128,128)
    if model1_name in ['srcnn', 'highresnet']:
        model1_im = cv2.resize(model1_im, (128,128))
    if model2_name in ['srcnn', 'highresnet']:
        model2_im = cv2.resize(model2_im, (128,128))

    # PSNR & SSIM
    psnr_model1 = calculate_psnr(naip_im, model1_im, 0)
    ssim_model1 = calculate_ssim(naip_im, model1_im, 0)
    psnr_model2 = calculate_psnr(naip_im, model2_im, 0)
    ssim_model2 = calculate_ssim(naip_im, model2_im, 0)
    if ( avg_human < 1 ) == (psnr_model1 > psnr_model2):
        correct['psnr'] += 1
    if ( avg_human < 1 ) == (ssim_model1 > ssim_model2):
        correct['ssim'] += 1

    # LPIPS
    normalized_naip = 2*(naip_im - np.amin(naip_im)) / (np.amax(naip_im) - np.amin(naip_im))-1
    naip_tensor = torch.tensor(np.transpose(normalized_naip, (2, 0, 1))).unsqueeze(0).float().to(device)
    normalized_m1 = 2*(model1_im - np.amin(model1_im)) / (np.amax(model1_im) - np.amin(model1_im))-1
    m1_tensor = torch.tensor(np.transpose(normalized_m1, (2, 0, 1))).unsqueeze(0).float().to(device)
    normalized_m2 = 2*(model2_im - np.amin(model2_im)) / (np.amax(model2_im) - np.amin(model2_im))-1
    m2_tensor = torch.tensor(np.transpose(normalized_m2, (2, 0, 1))).unsqueeze(0).float().to(device)
    alex_model1 = loss_fn_alex(naip_tensor, m1_tensor).detach().item()
    vgg_model1 = loss_fn_vgg(naip_tensor, m1_tensor).detach().item()
    alex_model2 = loss_fn_alex(naip_tensor, m2_tensor).detach().item()
    vgg_model2 = loss_fn_vgg(naip_tensor, m2_tensor).detach().item()
    if ( avg_human < 1 ) == (alex_model1 > alex_model2):
        correct['lpips_alex'] += 1
    if ( avg_human < 1 ) == (vgg_model1 > vgg_model2):
        correct['lpips_vgg'] += 1

    # CLIP
    naip_tensor2 = preprocess(Image.open(naip_fp)).unsqueeze(0).to(device)
    naip_feats = actual_clip_model.encode_image(naip_tensor2)
    m1_tensor2 = preprocess(Image.open(model1_fp)).unsqueeze(0).to(device)
    m1_feats = actual_clip_model.encode_image(m1_tensor2)
    m2_tensor2 = preprocess(Image.open(model2_fp)).unsqueeze(0).to(device)
    m2_feats = actual_clip_model.encode_image(m2_tensor2)
    clip_m1 = F.cosine_similarity(naip_feats, m1_feats).detach().item()
    clip_m2 = F.cosine_similarity(naip_feats, m2_feats).detach().item()
    if ( avg_human < 1 ) == (clip_m1 > clip_m2):
        correct['clip'] += 1

    # NAIP-CLIP 
    # these tensors are pulled from LPIPS code (normalized the same way)
    naip_feat = naip_clip_model.encode_image(naip_tensor)
    m1_feat = naip_clip_model.encode_image(m1_tensor)  
    m2_feat = naip_clip_model.encode_image(m2_tensor)
    naip_clip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
    naip_clip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
    if ( avg_human < 1 ) == (naip_clip_m1 > naip_clip_m2):
        correct['naip_clip'] += 1

    # SigLIP
    naip_pil = siglip_transforms(Image.open(naip_fp)).unsqueeze(0).to(device)
    m1_pil = siglip_transforms(Image.open(model1_fp)).unsqueeze(0).to(device)
    m2_pil = siglip_transforms(Image.open(model2_fp)).unsqueeze(0).to(device)
    naip_feat = siglip_model(naip_pil)
    m1_feat = siglip_model(m1_pil)
    m2_feat = siglip_model(m2_pil)
    siglip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
    siglip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
    if ( avg_human < 1 ) == (siglip_m1 > siglip_m2):
        correct['siglip'] += 1

    # Dino-v2
    # these tensors are pulled from LPIPS code (normalized the same way)
    naip_resized = torch.nn.functional.interpolate(naip_tensor, (126,126)).to(device)
    m1_resized = torch.nn.functional.interpolate(m1_tensor, (126,126)).to(device)
    m2_resized = torch.nn.functional.interpolate(m2_tensor, (126,126)).to(device)
    naip_feat = dinov2_vitg14(naip_resized)
    m1_feat = dinov2_vitg14(m1_resized)
    m2_feat = dinov2_vitg14(m2_resized)
    dino_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
    dino_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
    if ( avg_human < 1 ) == (dino_m1 > dino_m2):
        correct['dino'] += 1

    # Satlas Backbone
    # these tensors are pulled from LPIPS code (normalized the same way)
    naip_bck = satlas_backbone(naip_tensor)
    m1_bck = satlas_backbone(m1_tensor)
    m2_bck = satlas_backbone(m2_tensor)
    satbck_m1 = torch.mean(F.cosine_similarity(naip_bck[0], m1_bck[0])).detach().item()
    satbck_m2 = torch.mean(F.cosine_similarity(naip_bck[0], m2_bck[0])).detach().item()
    if ( avg_human < 1 ) == (satbck_m1 > satbck_m2):
        correct['satlas_backbone'] += 1

    # Satlas Intermediates
    naip_int = satlas_intermediates(naip_bck)
    m1_int = satlas_intermediates(m1_bck)
    m2_int = satlas_intermediates(m2_bck)
    satint_m1 = torch.mean(F.cosine_similarity(naip_int[0], m1_int[0])).detach().item()
    satint_m2 = torch.mean(F.cosine_similarity(naip_int[0], m2_int[0])).detach().item()
    if ( avg_human < 1 ) == (satint_m1 > satint_m2):
        correct['satlas_fpn'] += 1

print(correct)
