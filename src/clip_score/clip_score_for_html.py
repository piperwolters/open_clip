import os
import sys
import cv2
import glob
#import clip
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

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize pretrained super-res model. Load in weights.
sr_weights_path = '/data/piperw/super-res/satlas-super-resolution/experiments/satlas32_baseline/models/net_g_905000.pth'
sr_weights = torch.load(sr_weights_path)
sr_model = RRDBNet(num_in_ch=24, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=64, scale=4).to(device)
sr_model.load_state_dict(sr_weights['params_ema'])
sr_model.eval()

#### This loop is specific to the html vis - uses the folder compare/ containing numbered examples with various outputs in each ####
d = 'compare/'
for i in range(0, 30):
    base_path = d + str(i)
    if not os.path.exists(base_path):
        continue

    """
    Following code gets similarity scores using normal CLIP, between NAIP and Super-Res features
    """
    # Load, preprocess, and feed images through actual CLIP, then compute similarity scores
    naip_im = preprocess(Image.open(base_path + '/hr.png')).unsqueeze(0).to(device)
    high_im = preprocess(Image.open(base_path + '/highresnet.png')).unsqueeze(0).to(device)
    sr3_im = preprocess(Image.open(base_path + '/sr3_cfg.png')).unsqueeze(0).to(device)
    gan_im = preprocess(Image.open(base_path + '/esrgan.png')).unsqueeze(0).to(device)

    naip_feats = actual_clip_model.encode_image(naip_im)
    high_feats = actual_clip_model.encode_image(high_im)
    sr3_feats = actual_clip_model.encode_image(sr3_im)
    gan_feats = actual_clip_model.encode_image(gan_im)
   
    clip_naip_gan = str(F.cosine_similarity(naip_feats, gan_feats).detach().item())
    clip_naip_sr3 = str(F.cosine_similarity(naip_feats, sr3_feats).detach().item())
    clip_naip_high = str(F.cosine_similarity(naip_feats, high_feats).detach().item())

    # Divide each of these into 4x4 grids and then feed through model and take cosine similarity
    div = 4  # 4, 8, 16
    whl = 128
    l = int(whl / div)

    highs = np.zeros((div,div))
    sr3s = np.zeros((div,div))
    gans = np.zeros((div,div))
    for i in range(div):
        for j in range(div):
            naip_chunk = F.upsample(naip_im[:, :, i*l:(i+1)*l, j*l:(j+1)*l], (224,224))
            naip_feat = actual_clip_model.encode_image(naip_chunk)

            high_chunk = F.upsample(high_im[:, :, i*l:(i+1)*l, j*l:(j+1)*l], (224,224))
            high_feat = actual_clip_model.encode_image(high_chunk)

            sr3_chunk = F.upsample(sr3_im[:, :, i*l:(i+1)*l, j*l:(j+1)*l], (224,224))
            sr3_feat = actual_clip_model.encode_image(sr3_chunk)

            gan_chunk = F.upsample(gan_im[:, :, i*l:(i+1)*l, j*l:(j+1)*l], (224,224))
            gan_feat = actual_clip_model.encode_image(gan_chunk)

            naip_high_i_j = F.cosine_similarity(naip_feat, high_feat).detach().cpu()
            naip_sr3_i_j = F.cosine_similarity(naip_feat, sr3_feat).detach().cpu()
            naip_gan_i_j = F.cosine_similarity(naip_feat, gan_feat).detach().cpu()

            highs[i,j] = naip_high_i_j
            sr3s[i,j] = naip_sr3_i_j
            gans[i,j] = naip_gan_i_j
     
    # Plot the similarity scores of grid cells using matplotlib. 
    plt.imshow(highs, cmap='seismic', vmin=0.9, vmax=1.0)
    plt.colorbar()
    #plt.savefig(base_path+'/highs_16x16.png')
    plt.savefig(base_path+'/highs.png')
    plt.close()

    plt.imshow(sr3s, cmap='seismic', vmin=0.9, vmax=1.0)
    plt.colorbar()
    #plt.savefig(base_path+'/sr3s_16x16.png')
    plt.savefig(base_path+'/sr3s.png')
    plt.close()

    plt.imshow(gans, cmap='seismic', vmin=0.9, vmax=1.0)
    plt.colorbar()
    #plt.savefig(base_path+'/gans_16x16.png')
    plt.savefig(base_path+'/gan.png')
    plt.close()

    #skimage.io.imsave(base_path + '/highs.png', highs)
    #skimage.io.imsave(base_path + '/sr3s.png', sr3s)
    #skimage.io.imsave(base_path + '/gans.png', gans)

    """
    Following code gets similarity scores using SAT-CLIP, between Sentinel-2 and High-Res/Super-Res features
    """
    # Load all the images and pre-computed outputs
    s2 = cv2.resize(skimage.io.imread(base_path + '/s2.png'), (32, 32)) / 255 
    naip = skimage.io.imread(base_path + '/hr.png') /255
    high = skimage.io.imread(base_path + '/highresnet.png') /255
    sr3 = skimage.io.imread(base_path + '/sr3_cfg.png') / 255
    gan = skimage.io.imread(base_path + '/esrgan.png') / 255

    # Turn them all into tensors of the correct shape
    s2 = torch.tensor(np.transpose(s2, (2, 0, 1))).unsqueeze(0).float().to(device) 
    naip = torch.tensor(np.transpose(naip, (2, 0, 1))).unsqueeze(0).float().to(device) 
    high = torch.tensor(np.transpose(high, (2, 0, 1))).unsqueeze(0).float().to(device) 
    sr3 = torch.tensor(np.transpose(sr3, (2, 0, 1))).unsqueeze(0).float().to(device) 
    gan = torch.tensor(np.transpose(gan, (2, 0, 1))).unsqueeze(0).float().to(device)

    # Feed all these through their respective encoders of CLIP
    s2_emb = clip_model.encode_s2(s2)
    naip_emb = clip_model.encode_naip(naip)
    high_emb = clip_model.encode_naip(high)
    sr3_emb = clip_model.encode_naip(sr3)
    gan_emb = clip_model.encode_naip(gan)

    s2_naip = str(F.cosine_similarity(s2_emb, naip_emb).detach().item())
    s2_high = str(F.cosine_similarity(s2_emb, high_emb).detach().item())
    s2_sr3 = str(F.cosine_similarity(s2_emb, sr3_emb).detach().item())
    s2_gan = str(F.cosine_similarity(s2_emb, gan_emb).detach().item())

    with open(base_path + '/cos_sims.txt', 'w') as f:
        f.write(s2_naip+', '+s2_high+', '+s2_sr3+', '+s2_gan+', '+clip_s2_high+', '+clip_s2_sr3+', '+clip_s2_gan)

