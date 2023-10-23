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
from basicsr.archs.rrdbnet_arch import RRDBNet

sys.path.append('/data/piperw/open_clip/src/')
from infer_utils import sr_infer
from open_clip.model import CLIP, CLIPVisionCfg

device = torch.device('cpu')

# Organize the data that we will be using as input to various checkpoints of SR and SAT-CLIP model.
naip_dir = '/data/piperw/data/small_held_out_set/naip_128/'
naip_pngs = glob.glob(naip_dir + "*/tci/*.png")
s2_dir = '/data/piperw/data/small_held_out_set/s2_condensed/'
print("Using data set with ", len(naip_pngs), " datapoints.")

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
    for sr_chkpt in range(100000, 1000000, 100000):
        if not sr_chkpt in results[sc_chkpt]:
            results[sc_chkpt][sr_chkpt] = 0.0

        sr_weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_' + str(sr_chkpt) + '.pth'
        sr_weights = torch.load(sr_weights_path, map_location=torch.device('cpu'))
        sr_model.load_state_dict(sr_weights['params_ema'])
        sr_model.eval()

        # Now iterate over our test set, generating features and computing similarity scores.
        print("Using...", weights_path, " & ", sr_weights_path)
        idx_results = []
        for idx,naip_path in enumerate(naip_pngs):
            # Read in the NAIP image, aka the target
            naip_im = skimage.io.imread(naip_path)
            naip_im = np.transpose(naip_im, (2, 0, 1))
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
            sr_clip = sat_clip_model.encode_naip(sr_output)

            # Feed NAIP image through NAIP encoder of CLIP model
            naip_clip = sat_clip_model.encode_naip(naip_im)

            # Cosine similarity between SR and NAIP
            cos_naip_sr = F.cosine_similarity(naip_clip, sr_clip)

            idx_results.append(cos_naip_sr)

        # Take average over the test set scores
        results[sc_chkpt][sr_chkpt] = (sum(idx_results) / len(idx_results)).item()
        print("avg for idx ", idx, " : ", results[sc_chkpt][sr_chkpt])


# Once all runs have been computed, we want to save this dict to a json
# and experiment with visualizing it as plots.
import json
with open('diff_chkpt_results.json', 'w') as fp:
    json.dump(results, fp)
