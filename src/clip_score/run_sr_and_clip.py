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


device = torch.device('cuda')

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
        )

# Load the pretrained SAT-CLIP checkpoint. Load weights into model.
weights_path = '/data/piperw/open_clip/src/logs/8gpus-1s2-1024batch-2workers-1e4lr-newvalset/checkpoints/epoch_16.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
new_state_dict = {}
for k,v in state_dict.items():
    new_state_dict[k.replace('module.', '')] = v
sat_clip_model.load_state_dict(new_state_dict)
sat_clip_model.eval()
sat_clip_model.to(device)

# Initialize pretrained super-res model. Load in weights.
sr_weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_905000.pth'
sr_weights = torch.load(sr_weights_path)
sr_model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(device)
sr_model.load_state_dict(sr_weights['params_ema'])
sr_model.eval()

# Data paths
data_dir = '/data/piperw/data/small_held_out_set/'
s2_dir = data_dir + 's2_condensed/'
naip_dir = data_dir + 'naip_128/'
save_path = 'sr_outputs/'

# Following loop is for running both SR and CLIP on some val set.
# Run datapoints through super-res model and save outputs. 
naip_pngs = glob.glob(naip_dir + "*/tci/*.png")
print("Running inference on ", len(naip_pngs), " images.")
for idx,png in enumerate(naip_pngs):
    print("png:", png)

    # Extract chip and tile info from NAIP filepath
    split_path = png.split('/')
    chip = split_path[-1][:-4]
    tile = int(chip.split('_')[0]) // 16, int(chip.split('_')[1]) // 16
    s2_left_corner = tile[0] * 16, tile[1] * 16
    diffs = int(chip.split('_')[0]) - s2_left_corner[0], int(chip.split('_')[1]) - s2_left_corner[1]

    s2_path = os.path.join(s2_dir, str(tile[0])+'_'+str(tile[1]), str(diffs[1])+'_'+str(diffs[0])+'.png')
    naip_path = png

    save_dir = os.path.join(save_path, str(tile[0])+'_'+str(tile[1]))
    save_fn = save_dir + '/' + str(idx)
    os.makedirs(save_dir, exist_ok=True)

    naip_im = skimage.io.imread(naip_path)
    naip_im = np.transpose(naip_im, (2, 0, 1))
    naip_im = torch.tensor(naip_im).unsqueeze(0).float().to(device)

    s2_ims = skimage.io.imread(s2_path)

    # Feed S2 images through SR model.
    sr_output, s2_ims_tensor = sr_infer(sr_model, s2_ims, 8, device)
    sr_output.to(device)

    # Feed SR output through NAIP encoder of CLIP model
    sr_clip = sat_clip_model.encode_naip(sr_output)

    # Feed NAIP image through NAIP encoder of CLIP model
    naip_clip = sat_clip_model.encode_naip(naip_im)

    # Feed S2 images through S2 encoder of CLIP model
    s2_clip = sat_clip_model.encode_s2(s2_ims_tensor)

    # Cosine similarity between SR and S2
    cos_s2_sr = F.cosine_similarity(s2_clip, sr_clip)

    # Cosine similarity between NAIP and S2
    cos_s2_naip = F.cosine_similarity(s2_clip, naip_clip)

    #output = output.squeeze().cpu().detach().numpy()
    #output = np.transpose(output*255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
    #skimage.io.imsave(save_fn, output, check_contrast=False)

    break
