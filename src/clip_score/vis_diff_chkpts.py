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
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet

from metrics import *

sys.path.append('/data/piperw/open_clip_naipnaip/open_clip/src/')
from infer_utils import sr_infer
from open_clip.model import CLIP, CLIPVisionCfg

sys.path.append('/data/piperw/satlas-projects/satlas/')
from satlas.model.model import Model


device = torch.device('cuda')

# Organize the data that we will be using as input to various checkpoints of SR and SAT-CLIP model.
naip_dir = '/data/piperw/data/small_held_out_set/naip_128/'
naip_pngs = glob.glob(naip_dir + "*/tci/*.png")
s2_dir = '/data/piperw/data/small_held_out_set/s2_condensed/'

"""
# SAT-CLIP model config info.
s2_info = {}
s2_vision_cfg = CLIPVisionCfg(**s2_info)
naip_info = {}
naip_vision_cfg = CLIPVisionCfg(**naip_info)

# Initialize the SAT-CLIP model.
sat_clip_model = CLIP(
            embed_dim=512, 
            s2_vision_cfg=s2_vision_cfg, 
            naip_vision_cfg=naip_vision_cfg,
        ).to(device)
"""

# Intialize the Super-Res model. In this case, using ESRGAN.
sr_model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(device)

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


# Big dict of results
results = {}

# Iterate over each of the SAT-clip training checkpoints.
"""
for sc_chkpt in range(1, 16):
    if not str(sc_chkpt) in results:
        results[sc_chkpt] = {}

    weights_path = '/data/piperw/open_clip/src/logs/8gpus-1s2-1024batch-2workers-1e4lr-newvalset/checkpoints/epoch_' + str(sc_chkpt) + '.pt'
    weights_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    state_dict = weights_dict['state_dict']
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    sat_clip_model.load_state_dict(new_state_dict)
    sat_clip_model.eval()
"""
for i in range(1):

    # Iterate over each of the SR model checkpoints.
    for sr_chkpt in [5000, 10000, 15000, 25000, 50000, 100000, 250000, 500000, 905000]:

        #if not sr_chkpt in results[sc_chkpt]:
        #results[sc_chkpt][sr_chkpt] = 0.0
        results[sr_chkpt] = 0.0

        # Load in the ESRGAN super-res weights for the current iteration
        sr_weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_' + str(sr_chkpt) + '.pth'
        sr_weights = torch.load(sr_weights_path)
        sr_model.load_state_dict(sr_weights['params_ema'])
        sr_model.eval()

        # Now iterate over our test set, generating features and computing similarity scores.
        idx_results = []
        psnrs = []
        ssims = []
        clip_results = []
        naip_clip = []
        lpips_alex = []
        lpips_vgg = []
        siglip = []
        dino = []
        satlas_bck = []
        satlas_inter = []
        for idx,naip_path in enumerate(naip_pngs):
            # Read in the NAIP image, aka the target
            naip_orig_im = skimage.io.imread(naip_path)
            naip_im = np.transpose(naip_orig_im, (2, 0, 1))
            naip_tensor = torch.tensor(naip_im).unsqueeze(0).float().to(device)

            # Extract chip and tile info from NAIP filepath, and load S2 data
            split_path = naip_path.split('/')
            chip = split_path[-1][:-4]
            tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16
            s2_left_corner = tile[0] * 16, tile[1] * 16
            diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]
            s2_path = os.path.join(s2_dir, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')
            s2_ims = skimage.io.imread(s2_path)

            # Feed S2 images through SR model.
            sr_output, s2_ims_tensor = sr_infer(sr_model, s2_ims, 8, device)
            sr_output.to(device)

            """
            # SAT-CLIP trained on S2 and NAIP
            # Feed SR output through NAIP encoder of CLIP model
            sr_clip = sat_clip_model.encode_naip(sr_output * 255)
            # Feed NAIP image through NAIP encoder of CLIP model
            naip_clip = sat_clip_model.encode_naip(naip_im)
            # Cosine similarity between SR and NAIP
            cos_naip_sr = F.cosine_similarity(naip_clip, sr_clip)
            """

            # PSNR & SSIM
            sr = torch.permute(sr_output.squeeze(), (1, 2, 0)).detach().cpu().numpy() * 255
            naip = np.transpose(naip_im, (2, 1, 0))
            psnr = calculate_psnr(naip, sr, crop_border=0)
            ssim = calculate_ssim(naip, sr, crop_border=0)
            psnrs.append(psnr)
            ssims.append(ssim)

            # LPIPS
            normalized_naip = 2*(naip_im - np.amin(naip_im)) / (np.amax(naip_im) - np.amin(naip_im))-1
            normalized_naip = torch.tensor(normalized_naip).unsqueeze(0).float().to(device)
            normalized_sr = 2*(sr - np.amin(sr)) / (np.amax(sr) - np.amin(sr))-1
            normalized_sr = torch.permute(torch.tensor(normalized_sr), (2, 1, 0)).unsqueeze(0).float().to(device)
            alex = loss_fn_alex(normalized_naip, normalized_sr)
            lpips_alex.append(alex.detach().item())
            vgg = loss_fn_vgg(normalized_naip, normalized_sr)
            lpips_vgg.append(vgg.detach().item())

            # CLIP
            naip = preprocess(Image.open(naip_path)).unsqueeze(0).to(device)
            naip_feats = actual_clip_model.encode_image(naip)
            sr = torch.nn.functional.interpolate(sr_output, (224, 224)).to(device)
            sr_feats = actual_clip_model.encode_image(sr)
            sim = F.cosine_similarity(naip_feats, sr_feats).detach().item()
            clip_results.append(sim)

            # NAIP-CLIP 
            sr_feat = naip_clip_model.encode_image(sr_output)
            naip_feat = naip_clip_model.encode_image(naip_tensor)
            sim = F.cosine_similarity(naip_feat, sr_feat).detach().item()
            naip_clip.append(sim)

            # SigLIP
            naip_pil = Image.open(naip_path)
            naip_siglip = siglip_transforms(naip_pil).unsqueeze(0).to(device)
            naip_feat = siglip_model(naip_siglip)
            sr = torch.nn.functional.interpolate(sr_output, (224, 224)).to(device)
            sr_feat = siglip_model(sr)
            sim = F.cosine_similarity(naip_feat, sr_feat).detach().item()
            siglip.append(sim)

            # Dino-v2
            naip = torch.nn.functional.interpolate(normalized_naip, (126,126)).to(device)
            sr = torch.nn.functional.interpolate(normalized_sr, (126,126)).to(device)
            naip_feat = dinov2_vitg14(naip)
            sr_feat = dinov2_vitg14(sr)
            sim = F.cosine_similarity(naip_feat, sr_feat).detach().item()
            dino.append(sim)

            # Satlas Backbone
            naip_bck = satlas_backbone(normalized_naip)
            sr_bck = satlas_backbone(normalized_sr)
            sim = torch.mean(F.cosine_similarity(naip_bck[0], sr_bck[0])).detach().item()
            satlas_bck.append(sim)

            # Satlas Intermediates
            naip_int = satlas_intermediates(naip_bck)
            sr_int = satlas_intermediates(sr_bck)
            sim = torch.mean(F.cosine_similarity(naip_int[0], sr_int[0])).detach().item()
            satlas_inter.append(sim)

        # Take average over the test set scores and metrics
        #results[sr_chkpt] = [(sum(idx_results) / len(idx_results)).item()] #, sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)]
        print(sum(satlas_inter) / len(satlas_inter))


# Once all runs have been computed, we want to save this dict to a json
# and experiment with visualizing it as plots, in case something errors.
with open('diff_chkpt_results.json', 'w') as fp:
    json.dump(results, fp)
