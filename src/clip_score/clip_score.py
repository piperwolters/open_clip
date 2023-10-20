import os
import sys
import cv2
import glob
import clip
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

# CLIP model config info.
s2_info = {}
s2_vision_cfg = CLIPVisionCfg(**s2_info)
naip_info = {}
naip_vision_cfg = CLIPVisionCfg(**naip_info)
# Initialize the CLIP model.
clip_model = CLIP(
            embed_dim=512, 
            s2_vision_cfg=s2_vision_cfg, 
            naip_vision_cfg=naip_vision_cfg,
        )
# Load the pretrained CLIP checkpoint. Load weights into model.
weights_path = '/data/piperw/open_clip/src/logs/late-1mildataset-1gpu-1024batch-8workers/checkpoints/epoch_46.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
clip_model.load_state_dict(state_dict)
clip_model.eval()
clip_model.to(device)

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

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

#### This loop is specific to the html vis ####
d = 'compare/'
for i in range(0, 30):
    base_path = d + str(i)
    if not os.path.exists(base_path):
        continue

    # Load, preprocess, and feed images through actual CLIP
    naip_im = preprocess(Image.open(base_path + '/hr.png')).unsqueeze(0).to(device)
    high_im = preprocess(Image.open(base_path + '/highresnet.png')).unsqueeze(0).to(device)
    sr3_im = preprocess(Image.open(base_path + '/sr3_cfg.png')).unsqueeze(0).to(device)
    gan_im = preprocess(Image.open(base_path + '/esrgan.png')).unsqueeze(0).to(device)

    """
    naip_feats = actual_clip_model.encode_image(naip_im)
    high_feats = actual_clip_model.encode_image(high_im)
    sr3_feats = actual_clip_model.encode_image(sr3_im)
    gan_feats = actual_clip_model.encode_image(gan_im)
    """

    # Divide each of these into 4x4 grids and then feed through model and take cosine similarity
    div = 4  # 4
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

    continue

    """
    clip_s2_high = str(F.cosine_similarity(naip_feats, high_feats).detach().item())
    clip_s2_sr3 = str(F.cosine_similarity(naip_feats, sr3_feats).detach().item())
    clip_s2_gan = str(F.cosine_similarity(naip_feats, gan_feats).detach().item())
    print("clip results:", clip_s2_high, clip_s2_sr3, clip_s2_gan)

    # Load all the images and pre-computed outputs
    s2 = cv2.resize(skimage.io.imread(base_path + '/s2.png'), (32, 32))
    naip = skimage.io.imread(base_path + '/hr.png')
    high = skimage.io.imread(base_path + '/highresnet.png')
    sr3 = skimage.io.imread(base_path + '/sr3_cfg.png')
    gan = skimage.io.imread(base_path + '/esrgan.png')

    # Turn them all into tensors of the correct shape
    s2 = torch.tensor(np.transpose(s2, (2, 0, 1))).unsqueeze(0).float().to(device)/255
    naip = torch.tensor(np.transpose(naip, (2, 0, 1))).unsqueeze(0).float().to(device)/255
    high = torch.tensor(np.transpose(high, (2, 0, 1))).unsqueeze(0).float().to(device)/255
    sr3 = torch.tensor(np.transpose(sr3, (2, 0, 1))).unsqueeze(0).float().to(device)/255
    gan = torch.tensor(np.transpose(gan, (2, 0, 1))).unsqueeze(0).float().to(device)/255
    print("tensor shapes:", s2.shape, naip.shape, high.shape, sr3.shape, gan.shape)

    # Feed all these through their respective encoders of CLIP
    s2_emb = clip_model.encode_s2(s2)
    naip_emb = clip_model.encode_naip(naip)
    high_emb = clip_model.encode_naip(high)
    sr3_emb = clip_model.encode_naip(sr3)
    gan_emb = clip_model.encode_naip(gan)
    print("output shapes:", s2_emb.shape, naip_emb.shape, high_emb.shape, sr3_emb.shape, gan_emb.shape)

    s2_naip = str(F.cosine_similarity(s2_emb, naip_emb).detach().item())
    s2_high = str(F.cosine_similarity(s2_emb, high_emb).detach().item())
    s2_sr3 = str(F.cosine_similarity(s2_emb, sr3_emb).detach().item())
    s2_gan = str(F.cosine_similarity(s2_emb, gan_emb).detach().item())
    print("results:", s2_naip, s2_high, s2_sr3, s2_gan)

    with open(base_path + '/cos_sims.txt', 'w') as f:
        f.write(s2_naip+', '+s2_high+', '+s2_sr3+', '+s2_gan+', '+clip_s2_high+', '+clip_s2_sr3+', '+clip_s2_gan)
    """

exit()
########

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
    print(s2_path)
    print(naip_path)

    save_dir = os.path.join(save_path, str(tile[0])+'_'+str(tile[1]))
    save_fn = save_dir + '/' + str(idx)
    os.makedirs(save_dir, exist_ok=True)

    naip_im = skimage.io.imread(naip_path)
    print(naip_im.shape)
    naip_im = np.transpose(naip_im, (2, 0, 1))
    naip_im = torch.tensor(naip_im).unsqueeze(0).float().to(device)

    s2_ims = skimage.io.imread(s2_path)
    print("s2 im:", s2_ims.shape)

    # Feed S2 images through SR model.
    sr_output, s2_ims_tensor = sr_infer(sr_model, s2_ims, 8, device)
    sr_output.to(device)
    print("sr_output:", sr_output.shape)
    print("s2 ims tensor:", s2_ims_tensor.shape)

    # Feed SR output through NAIP encoder of CLIP model
    sr_clip = clip_model.encode_naip(sr_output)
    print("sr_clip:", sr_clip.shape)

    # Feed NAIP image through NAIP encoder of CLIP model
    naip_clip = clip_model.encode_naip(naip_im)

    # Feed S2 images through S2 encoder of CLIP model
    s2_clip = clip_model.encode_s2(s2_ims_tensor)
    print("s2 clip:", s2_clip.shape)

    # Cosine similarity between SR and S2
    cos_s2_sr = F.cosine_similarity(s2_clip, sr_clip)
    print("cos_s2_sr:", cos_s2_sr)

    # Cosine similarity between NAIP and S2
    cos_s2_naip = F.cosine_similarity(s2_clip, naip_clip)
    print("cos_s2_naip:", cos_s2_naip)

    #output = output.squeeze().cpu().detach().numpy()
    #output = np.transpose(output*255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
    #skimage.io.imsave(save_fn, output, check_contrast=False)

    break
