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
from open_clip.model import CLIP, CLIPVisionCfg
import tqdm

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
#weights_path = '/data/piperw/open_clip/src/logs/8gpus-1s2-1024batch-2workers-1e4lr-newvalset/checkpoints/epoch_16.pt'
#weights_path = '/home/favyenb/open_clip/src/logs/4gpus-1s2-1024batch-2workers-1e4lr-newvalset-urban/checkpoints/epoch_323.pt'
weights_path = '/home/favyenb/open_clip/src/logs/8gpus-1s2-2048batch-2workers-1e4lr-newvalset-urban/checkpoints/epoch_227.pt'
weights_dict = torch.load(weights_path)
state_dict = weights_dict['state_dict']
state_dict = {k[7:]: v for k, v in state_dict.items()}
clip_model.load_state_dict(state_dict)
clip_model.eval()
clip_model.to(device)

# Load pretrained CLIP (normal CLIP)
actual_clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Build groups of NAIP images.
tile_to_fnames = {}
img_dir = '/data/favyenb/naip_for_clip_check_sm/'
for image_id in ['v1', 'v2', 'v3']:
    tci_dir = os.path.join(img_dir, image_id, 'tci')
    for fname in os.listdir(tci_dir):
        parts = fname.split('.')[0].split('_')
        tile = (int(parts[0]), int(parts[1]))
        if tile not in tile_to_fnames:
            tile_to_fnames[tile] = []
        tile_to_fnames[tile].append(os.path.join(tci_dir, fname))
# Delete groups with not enough images.
tile_to_fnames = {tile: fnames for tile, fnames in tile_to_fnames.items() if len(fnames) >= 3}

clip_close_scores = []
clip_far_scores = []
our_close_scores = []
our_far_scores = []

for tile, fnames in tqdm.tqdm(tile_to_fnames.items()):
    orig_ims = [skimage.io.imread(fname) for fname in fnames]
    orig_ims = [skimage.transform.resize(im, (128, 128), preserve_range=True).astype(np.uint8) for im in orig_ims]

    images = [preprocess(Image.fromarray(im)).unsqueeze(0).to(device) for im in orig_ims]
    embeddings = [actual_clip_model.encode_image(im) for im in images]
    close_sim = F.cosine_similarity(embeddings[1], embeddings[2]).detach().item()
    far_sim = F.cosine_similarity(embeddings[0], embeddings[2]).detach().item()
    clip_close_scores.append(close_sim)
    clip_far_scores.append(far_sim)

    images = [
        torch.tensor(np.transpose(im, (2, 0, 1))).unsqueeze(0).float().to(device)/255
        for im in orig_ims
    ]
    embeddings = [clip_model.encode_naip(im) for im in images]
    close_sim = F.cosine_similarity(embeddings[1], embeddings[2]).detach().item()
    far_sim = F.cosine_similarity(embeddings[0], embeddings[2]).detach().item()
    our_close_scores.append(close_sim)
    our_far_scores.append(far_sim)

print('clip_close={} +/- {}  clip_far={} +/- {}  our_close={} +/- {}  our_far={} +/- {}'.format(
    np.mean(clip_close_scores),
    np.std(clip_close_scores),
    np.mean(clip_far_scores),
    np.std(clip_far_scores),
    np.mean(our_close_scores),
    np.std(our_close_scores),
    np.mean(our_far_scores),
    np.std(our_far_scores),
))
print('clip_accuracy={} our_accuracy={}'.format(
    np.mean([int(clip_close_scores[i] > clip_far_scores[i]) for i in range(len(clip_close_scores))]),
    np.mean([int(our_close_scores[i] > our_far_scores[i]) for i in range(len(clip_close_scores))]),
))



'''
import os
import sys
import cv2
import glob
import skimage.io
import numpy as np
import tqdm

# Build groups of NAIP images.
tile_to_fnames = {}
img_dir = '/data/favyenb/naip_for_clip_check_sm/'
for image_id in ['v1', 'v2', 'v3']:
    tci_dir = os.path.join(img_dir, image_id, 'tci')
    for fname in os.listdir(tci_dir):
        parts = fname.split('.')[0].split('_')
        tile = (int(parts[0]), int(parts[1]))
        if tile not in tile_to_fnames:
            tile_to_fnames[tile] = []
        tile_to_fnames[tile].append(os.path.join(tci_dir, fname))
# Delete groups with not enough images.
tile_to_fnames = {tile: fnames for tile, fnames in tile_to_fnames.items() if len(fnames) >= 3}

accuracy = []

for tile, fnames in tqdm.tqdm(list(tile_to_fnames.items())):
    orig_ims = [skimage.io.imread(fname) for fname in fnames]
    orig_ims = [skimage.transform.resize(im, (128, 128), preserve_range=True).astype(np.uint8) for im in orig_ims]

    #close_score = np.mean(np.abs(orig_ims[1] - orig_ims[2]))
    #far_score = np.mean(np.abs(orig_ims[0] - orig_ims[2]))
    close_score = cv2.PSNR(orig_ims[1], orig_ims[2])
    far_score = cv2.PSNR(orig_ims[0], orig_ims[2])
    if close_score < far_score:
        accuracy.append(0)
    else:
        accuracy.append(1)

print(np.mean(accuracy))
'''