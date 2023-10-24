import os
import sys
import cv2
import json
import clip
import glob
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
sys.path.append('/data/piperw/open_clip/src/')
from infer_utils import sr_infer
from open_clip.model import CLIP, CLIPVisionCfg

device = torch.device('cpu')

# Organize the data that we will be using as input to various checkpoints of SR and SAT-CLIP model.
naip_dir = '/data/piperw/data/small_held_out_set/naip_128/'
naip_pngs = glob.glob(naip_dir + "*/tci/*.png")
s2_dir = '/data/piperw/data/small_held_out_set/s2_condensed/'

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

# Intialize the Super-Res model. In this case, using ESRGAN.
sr_model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(device)

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/16", device=device)

# Big dict of results
results = {}

# Iterate over each of the SAT-clip training checkpoints.
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

    # Iterate over each of the SR model checkpoints.
    for sr_chkpt in [5000, 10000, 15000, 25000, 50000, 100000, 250000, 500000, 905000]:
        if not sr_chkpt in results[sc_chkpt]:
            results[sc_chkpt][sr_chkpt] = 0.0

        sr_weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_' + str(sr_chkpt) + '.pth'
        sr_weights = torch.load(sr_weights_path, map_location=torch.device('cpu'))
        sr_model.load_state_dict(sr_weights['params_ema'])
        sr_model.eval()

        # Now iterate over our test set, generating features and computing similarity scores.
        #print("Using...", weights_path, " & ", sr_weights_path)
        idx_results = []
        psnrs = []
        ssims = []
        clip_results = []
        for idx,naip_path in enumerate(naip_pngs):
            # Read in the NAIP image, aka the target
            naip_orig_im = skimage.io.imread(naip_path)
            naip_im = np.transpose(naip_orig_im, (2, 0, 1))
            naip_im = torch.tensor(naip_im).unsqueeze(0).float().to(device)

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

            # Feed SR output through NAIP encoder of CLIP model
            sr_clip = sat_clip_model.encode_naip(sr_output * 255)

            # Feed NAIP image through NAIP encoder of CLIP model
            naip_clip = sat_clip_model.encode_naip(naip_im)

            # Cosine similarity between SR and NAIP
            cos_naip_sr = F.cosine_similarity(naip_clip, sr_clip)

            idx_results.append(cos_naip_sr)

            """
            # Also compute PSNR and SSIM between target and output
            sr_output = torch.permute(sr_output.squeeze(), (1, 2, 0)).detach().numpy() * 255
            naip_output = np.transpose(naip_im.squeeze(), (1, 2, 0)).detach().numpy()
            psnr = calculate_psnr(naip_output, sr_output, crop_border=0)
            ssim = calculate_ssim(naip_output, sr_output, crop_border=0)
            psnrs.append(psnr)
            ssims.append(ssim)

            # Actual CLIP feature generation and similarity score calculation 
            naip = preprocess(Image.open(naip_path)).unsqueeze(0).to(device)
            naip_feats = actual_clip_model.encode_image(naip)
            sr = torch.from_numpy(np.transpose(cv2.resize(sr_output/255, (224, 224)), (2, 1, 0))).unsqueeze(0).to(device)
            sr_feats = actual_clip_model.encode_image(sr)
            sim_score = F.cosine_similarity(naip_feats, sr_feats)
            clip_results.append(sim_score.item())
            """

        # Take average over the test set scores and metrics
        results[sc_chkpt][sr_chkpt] = [(sum(idx_results) / len(idx_results)).item()] #, sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)]

# Once all runs have been computed, we want to save this dict to a json
# and experiment with visualizing it as plots, in case something errors.
with open('diff_chkpt_results.json', 'w') as fp:
    json.dump(results, fp)

# Skip down to this visualization step if 1) the json is already created or 2) whole script is run end2end.
f = open('diff_chkpt_results.json')
data = json.load(f)
x_labels, y_labels, points = [], [], []
psnrs, ssims = [], []
for i,(sr_ckpt_k, sr_ckpt_v) in enumerate(data.items()):
    y_labels.append(sr_ckpt_k)
    for clip_ckpt_k, clip_ckpt_v in data[sr_ckpt_k].items():
        # Only make this list once since this loop will iterate for each sr_ckpt
        if i == 0:
            x_labels.append(clip_ckpt_k)

        print(data[sr_ckpt_k][clip_ckpt_k][1])
        points.append(data[sr_ckpt_k][clip_ckpt_k][0])
        psnrs.append(data[sr_ckpt_k][clip_ckpt_k][1])
        ssims.append(data[sr_ckpt_k][clip_ckpt_k][2])
