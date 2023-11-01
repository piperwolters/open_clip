import os
import sys
import cv2
import json
import torch
import skimage.io
import numpy as np
import torch.nn.functional as F
from statistics import mean
from PIL import Image
from metrics import *

device = torch.device('cuda')

application_type = 'natural'   # naip_s2, natural, faces, worldstrat

if application_type == 'natural':
    annots_file = open('/data/piperw/data/mturk/natural_images/mturk_natural_batch01_dict.json')
    annots = json.load(annots_file)
    data_dir = '/data/piperw/data/mturk/natural_images/natural_human_feedback_crop/'

    model_names = ['bicubic', 'edsr_r32f256', 'nina_b2', 'realesrgan_generalx4v3', 'edsr_baseline',
                'nina_b0', 'realesrgan', 'sr3distort']
elif application_type == 'naip_s2':
    # Using json file that contains annotations for BOTH batch01 and batch02
    # ex: {"36005_51983": {"model1": "esrgan_osm_chkpt50k", "model2": "srcnn", "answers": [0, 0, 0, 0, 0]}
    annots_file = open('/data/piperw/scripts/mturk_batch01and02_dict.json')
    annots = json.load(annots_file)

    # Data directory containing symlinks to both urban outputs and disjoint-from-training outputs
    data_dir = '/data/piperw/data/mturk/all_mturk_outputs/'

    model_names = ['srcnn','highresnet','sr3','sr3_cfg','esrgan_satlas','esrgan_satlas_chkpt5k',
                    'esrgan_satlas_chkpt50k','esrgan_osm','esrgan_osm_chkpt5k','esrgan_osm_chkpt50k']
else:
    print("Note implemented yet...")


# Specify list of metrics you want to run so that jobs can be dispursed onto different machines.
metrics2run = [
                #'psnr', 
                #'ssim', 
                #'cpsnr',
                #'lpips_alex', 
                #'lpips_vgg', 
                #'clip',
                #'naip_clip', 
                #'sat_clip',
                #'siglip', 
                #'dino', 
                #'satlas_backbone', 
                #'satlas_fpn',
                #'metaclip', 
                #'sam', 
                #'siglip_400m', 
                #'eva', 
                #'clipa', 
                #'eva_plus', 
                'another_siglip'
            ]

# TODO: deal with naip_clip, sat_clip, and models that use open_clip not playing nice,
# but until then just make sure to run those separately.
# Before running naip_clip or sat_clip make sure to pip uninstall open_clip_torch
if 'naip_clip' in metrics2run or 'sat_clip' in metrics2run:
    print("WARNING: did you uninstall open_clip_torch before running naip_clip or sat_clip?")

########################################################################################################

if 'metaclip' in metrics2run:
    # MetaCLIP (open_clip)
    import open_clip
    metaclip_model, _, metaclip_preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m')
    metaclip_model.to(device)

if 'sam' in metrics2run:
    # SAMScore evaluator
    import samscore
    SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_b" ).to(device)

if 'siglip_400m' in metrics2run:
    # Other SigLIP?
    from open_clip import create_model_from_pretrained
    siglip_model2, siglip_preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    siglip_model2.to(device)

if 'eva' in metrics2run:
    # EVA
    import timm
    eva_model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True, num_classes=0).to(device)
    data_config = timm.data.resolve_model_data_config(eva_model)
    eva_transforms = timm.data.create_transform(**data_config, is_training=False)

if 'clip' in metrics2run:
    # Load pretrained CLIP (normal CLIP)
    import clip
    actual_clip_model, preprocess = clip.load("ViT-B/16", device=device)

if 'clipa' in metrics2run:
    import open_clip
    # More open_clip models - https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
    clipa, _, clipa_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
    clipa.to(device)

if 'eva_plus' in metrics2run:
    import open_clip
    eva_plus, _, eva_plus_preprocess = open_clip.create_model_and_transforms('EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
    eva_plus.to(device)

if 'another_siglip' in metrics2run:
    import open_clip
    another_siglip, _, another_siglip_preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
    another_siglip.to(device)

if 'siglip' in metrics2run:
    import timm
    # Load pretrained SigLIP (using timm)
    siglip_model = timm.create_model(
        'vit_base_patch16_siglip_224',
        pretrained=True,
        num_classes=0,
    ).eval().to(device)
    data_config = timm.data.resolve_model_data_config(siglip_model)
    siglip_transforms = timm.data.create_transform(**data_config, is_training=False)

if 'lpips_alex' or 'lpips_vgg' in metrics2run:
    # LPIPS perceptual losses / scores
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

if 'dino' in metrics2run:
    # Initiatialize the DinoV2 model
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)


# NOTE: it is annoying to run the three different open_clip versions in one script

if 'sat_clip' in metrics2run:
    # SAT-CLIP
    sys.path.append('/data/piperw/open_clip/src/')
    from open_clip.model import CLIP as SATCLIP
    from open_clip.model import CLIPVisionCfg as SATCLIPVisionCfg
    # SAT-CLIP model config info.
    s2_info = {}
    s2_vision_cfg = SATCLIPVisionCfg(**s2_info)
    naip_info = {}
    naip_vision_cfg = SATCLIPVisionCfg(**naip_info)
    # Initialize the SAT-CLIP model.
    sat_clip_model = SATCLIP(
                embed_dim=512,
                s2_vision_cfg=s2_vision_cfg,
                naip_vision_cfg=naip_vision_cfg,
            ).to(device)
    # Load weights in
    weights_path = '/data/piperw/open_clip/src/logs/8gpus-1s2-1024batch-2workers-1e4lr-newvalset/checkpoints/epoch_26.pt'
    weights_dict = torch.load(weights_path, map_location=torch.device('cuda'))
    state_dict = weights_dict['state_dict']
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    sat_clip_model.load_state_dict(new_state_dict)
    sat_clip_model.eval()

if 'naip_clip' in metrics2run:
    # NAIP-NAIP-CLIP 
    sys.path.append('/data/piperw/open_clip_naipnaip/open_clip/src/')
    from infer_utils import sr_infer
    from open_clip.model import CLIP, CLIPVisionCfg
    # NAIP-CLIP model config info
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

if 'satlas_backbone' in metrics2run or 'satlas_fpn' in metrics2run:
    # Satlas model
    sys.path.append('/data/piperw/satlas-projects/satlas/')
    from satlas.model.model import Model
    with open('/data/piperw/satlas-projects/satlas/old_mi.txt', 'r') as f:
        satlas_cfg = json.load(f)
    satlas_model = Model({'config': satlas_cfg['Model'], 'channels': ['tci'], 'tasks': satlas_cfg['Tasks']}).to(device)
    satlas_model.load_state_dict(torch.load('/data/piperw/satlas-projects/satlas/satlas.pth'), strict=False)
    satlas_backbone = satlas_model.backbone
    satlas_intermediates = satlas_model.intermediates

########################################################################################################

metric_names = ['psnr', 'ssim', 'lpips_alex', 'lpips_vgg', 'clip', 'naip_clip', 'sat_clip', 'siglip', 'dino', 'satlas_backbone', 'satlas_fpn',
                'metaclip', 'sam', 'siglip_400m', 'eva', 'clipa', 'eva_plus', 'another_siglip', 'cpsnr']

correct = {mn: 0 for mn in metrics2run}
print("Running the metrics:", metrics2run)

counter = 0

print("Iterating through ", len(annots.items()), " datapoints.")
for idx,(chip, d) in enumerate(annots.items()):
    print("Processing...", idx)

    if 'ADE' in chip:
        print(chip)
        counter += 1
    else:
        continue

    human_answers = d['answers']
    avg_human = mean(human_answers)

    model1_name = d['model1']
    model2_name = d['model2']

    if application_type == 'naip_s2':
        naip_fp = data_dir + chip + '/' + 'naip.png'
    else:
        naip_fp = data_dir + chip + '/' + 'highres.png' 
    model1_fp = data_dir + chip + '/' + model1_name + '.png'
    model2_fp = data_dir + chip + '/' + model2_name + '.png'

    # Load the three images with skimage for metrics that want numpy arrays
    naip_im = skimage.io.imread(naip_fp)
    model1_im = skimage.io.imread(model1_fp)
    model2_im = skimage.io.imread(model2_fp)

    # Load the three images with PIL for metrics that particularly want this
    naip_im_pil = Image.open(naip_fp)
    m1_im_pil = Image.open(model1_fp)
    m2_im_pil = Image.open(model2_fp)

    # Account for SRCNN and HighResNet outputs being (640,640) instead of (128,128)
    if model1_name in ['srcnn', 'highresnet']:
        model1_im = cv2.resize(model1_im, (128,128))
        m1_im_pil = m1_im_pil.resize((128,128))
    if model2_name in ['srcnn', 'highresnet']:
        model2_im = cv2.resize(model2_im, (128,128))
        m2_im_pil = m2_im_pil.resize((128,128))

    # Normalized tensors that multiple metrics need
    normalized_naip = 2*(naip_im - np.amin(naip_im)) / (np.amax(naip_im) - np.amin(naip_im))-1
    naip_tensor = torch.tensor(np.transpose(normalized_naip, (2, 0, 1))).unsqueeze(0).float().to(device)
    normalized_m1 = 2*(model1_im - np.amin(model1_im)) / (np.amax(model1_im) - np.amin(model1_im))-1
    m1_tensor = torch.tensor(np.transpose(normalized_m1, (2, 0, 1))).unsqueeze(0).float().to(device)
    normalized_m2 = 2*(model2_im - np.amin(model2_im)) / (np.amax(model2_im) - np.amin(model2_im))-1
    m2_tensor = torch.tensor(np.transpose(normalized_m2, (2, 0, 1))).unsqueeze(0).float().to(device)
    
    if 'psnr' in metrics2run or 'ssim' in metrics2run:
        # PSNR & SSIM
        psnr_model1 = calculate_psnr(naip_im, model1_im, 0)
        ssim_model1 = calculate_ssim(naip_im, model1_im, 0)
        psnr_model2 = calculate_psnr(naip_im, model2_im, 0)
        ssim_model2 = calculate_ssim(naip_im, model2_im, 0)
        if ( avg_human < 1 ) == (psnr_model1 > psnr_model2):
            correct['psnr'] += 1
        if ( avg_human < 1 ) == (ssim_model1 > ssim_model2):
            correct['ssim'] += 1

    if 'cpsnr' in metrics2run:
        cpsnr_model1 = calculate_cpsnr(naip_im, model1_im, 3)
        cpsnr_model2 = calculate_cpsnr(naip_im, model2_im, 3)
        if ( avg_human < 1 ) == (cpsnr_model1 > cpsnr_model2):
            correct['cpsnr'] += 1

    if 'lpips_alex' in metrics2run or 'lpips_vgg' in metrics2run:
        # LPIPS
        alex_model1 = loss_fn_alex(naip_tensor, m1_tensor).detach().item()
        vgg_model1 = loss_fn_vgg(naip_tensor, m1_tensor).detach().item()
        alex_model2 = loss_fn_alex(naip_tensor, m2_tensor).detach().item()
        vgg_model2 = loss_fn_vgg(naip_tensor, m2_tensor).detach().item()
        if ( avg_human < 1 ) == (alex_model1 > alex_model2):
            correct['lpips_alex'] += 1
        if ( avg_human < 1 ) == (vgg_model1 > vgg_model2):
            correct['lpips_vgg'] += 1

    if 'clip' in metrics2run:
        # CLIP
        naip_tensor2 = preprocess(naip_im_pil).unsqueeze(0).to(device)
        naip_feats = actual_clip_model.encode_image(naip_tensor2)
        m1_tensor2 = preprocess(m1_im_pil).unsqueeze(0).to(device)
        m1_feats = actual_clip_model.encode_image(m1_tensor2)
        m2_tensor2 = preprocess(m2_im_pil).unsqueeze(0).to(device)
        m2_feats = actual_clip_model.encode_image(m2_tensor2)
        clip_m1 = F.cosine_similarity(naip_feats, m1_feats).detach().item()
        clip_m2 = F.cosine_similarity(naip_feats, m2_feats).detach().item()
        if ( avg_human < 1 ) == (clip_m1 > clip_m2):
            correct['clip'] += 1

    if 'clipa' in metrics2run:
        # CLIPA (open_clip: 'ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
        naip_clipa = clipa_preprocess(naip_im_pil).unsqueeze(0).to(device)
        m1_clipa = clipa_preprocess(m1_im_pil).unsqueeze(0).to(device)
        m2_clipa = clipa_preprocess(m2_im_pil).unsqueeze(0).to(device)
        naip_feats = clipa.encode_image(naip_clipa)
        m1_feats = clipa.encode_image(m1_clipa)
        m2_feats = clipa.encode_image(m2_clipa)
        clipa_m1 = F.cosine_similarity(naip_feats, m1_feats).detach().item()
        clipa_m2 = clip_m2 = F.cosine_similarity(naip_feats, m2_feats).detach().item()
        if ( avg_human < 1 ) == (clipa_m1 > clipa_m2):
            correct['clipa'] += 1

    if 'eva_plus' in metrics2run:
        # EVA (open_clip: 'EVA02-E-14-plus', pretrained='laion2b_s9b_b144k')
        naip_eva = eva_plus_preprocess(naip_im_pil).unsqueeze(0).to(device)
        m1_eva = eva_plus_preprocess(m1_im_pil).unsqueeze(0).to(device)
        m2_eva = eva_plus_preprocess(m2_im_pil).unsqueeze(0).to(device)
        naip_feats = eva_plus.encode_image(naip_eva)
        m1_feats = eva_plus.encode_image(m1_eva)
        m2_feats = eva_plus.encode_image(m2_eva)
        eva_m1 = F.cosine_similarity(naip_feats, m1_feats).detach().item()
        eva_m2 = clip_m2 = F.cosine_similarity(naip_feats, m2_feats).detach().item()
        if ( avg_human < 1 ) == (eva_m1 > eva_m2):
            correct['eva_plus'] += 1

    if 'another_siglip' in metrics2run:
        # SigLIP (open_clip: 'ViT-SO400M-14-SigLIP-384', pretrained='webli')
        naip_another_siglip = another_siglip_preprocess(naip_im_pil).unsqueeze(0).to(device)
        m1_another_siglip = another_siglip_preprocess(m1_im_pil).unsqueeze(0).to(device)
        m2_another_siglip = another_siglip_preprocess(m2_im_pil).unsqueeze(0).to(device)
        naip_feats = another_siglip.encode_image(naip_another_siglip)
        m1_feats = another_siglip.encode_image(m1_another_siglip)
        m2_feats = another_siglip.encode_image(m2_another_siglip)
        another_siglip_m1 = F.cosine_similarity(naip_feats, m1_feats).detach().item()
        another_siglip_m2 = clip_m2 = F.cosine_similarity(naip_feats, m2_feats).detach().item()
        if ( avg_human < 1 ) == (another_siglip_m1 > another_siglip_m2):
            correct['another_siglip'] += 1

    if 'naip_clip' in metrics2run:
        # NAIP-CLIP 
        # these tensors are pulled from LPIPS code (normalized the same way)
        naip_feat = naip_clip_model.encode_image(naip_tensor)
        m1_feat = naip_clip_model.encode_image(m1_tensor)  
        m2_feat = naip_clip_model.encode_image(m2_tensor)
        naip_clip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        naip_clip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (naip_clip_m1 > naip_clip_m2):
            correct['naip_clip'] += 1

    if 'sat_clip' in metrics2run:
        # SAT-CLIP trained on S2 and NAIP
        naip_feat = sat_clip_model.encode_naip(naip_tensor)
        m1_feat = sat_clip_model.encode_naip(m1_tensor)
        m2_feat = sat_clip_model.encode_naip(m2_tensor)
        sat_clip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        sat_clip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (sat_clip_m1 > sat_clip_m2):
            correct['sat_clip'] += 1

    if 'siglip' in metrics2run:
        # SigLIP
        naip_pil = siglip_transforms(naip_im_pil).unsqueeze(0).to(device)
        m1_pil = siglip_transforms(m1_im_pil).unsqueeze(0).to(device)
        m2_pil = siglip_transforms(m2_im_pil).unsqueeze(0).to(device)
        naip_feat = siglip_model(naip_pil)
        m1_feat = siglip_model(m1_pil)
        m2_feat = siglip_model(m2_pil)
        siglip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        siglip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (siglip_m1 > siglip_m2):
            correct['siglip'] += 1

    if 'dino' in metrics2run:
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

    if 'satlas_backbone' in metrics2run or 'satlas_fpn' in metrics2run:
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
    
    if 'siglip_400m' in metrics2run:
        # Bigger? SigLIP
        naip_pil = siglip_preprocess(naip_im_pil).unsqueeze(0).to(device)
        m1_pil = siglip_preprocess(m1_im_pil).unsqueeze(0).to(device)
        m2_pil = siglip_preprocess(m2_im_pil).unsqueeze(0).to(device)
        naip_feat = siglip_model2.encode_image(naip_pil)
        m1_feat = siglip_model2.encode_image(m1_pil)
        m2_feat = siglip_model2.encode_image(m2_pil)
        siglip2_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        siglip2_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (siglip2_m1 > siglip2_m2):
            correct['siglip_400m'] += 1

    if 'metaclip' in metrics2run:
        # MetaCLIP
        naip = metaclip_preprocess(naip_im_pil).unsqueeze(0).to(device)
        m1 = metaclip_preprocess(m1_im_pil).unsqueeze(0).to(device)
        m2 = metaclip_preprocess(m2_im_pil).unsqueeze(0).to(device)
        naip_feat = metaclip_model.encode_image(naip)
        m1_feat = metaclip_model.encode_image(m1)
        m2_feat = metaclip_model.encode_image(m2)
        metaclip_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        metaclip_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (metaclip_m1 > metaclip_m2):
            correct['metaclip'] += 1

    if 'eva' in metrics2run:
        # EVA
        naip = eva_transforms(naip_im_pil).unsqueeze(0).to(device)
        m1 = eva_transforms(m1_im_pil).unsqueeze(0).to(device)
        m2 = eva_transforms(m2_im_pil).unsqueeze(0).to(device)
        naip_feat = eva_model(naip)
        m1_feat = eva_model(m1)
        m2_feat = eva_model(m2)
        eva_m1 = F.cosine_similarity(naip_feat, m1_feat).detach().item()
        eva_m2 = F.cosine_similarity(naip_feat, m2_feat).detach().item()
        if ( avg_human < 1 ) == (eva_m1 > eva_m2):
            correct['eva'] += 1

    if 'sam' in metrics2run:
        # SAMScore
        m1_samscore = SAMScore_Evaluation.evaluation_from_path(source_image_path=naip_fp,  generated_image_path=model1_fp)
        m2_samscore = SAMScore_Evaluation.evaluation_from_path(source_image_path=naip_fp,  generated_image_path=model2_fp)
        if ( avg_human < 1 ) == (m1_samscore > m2_samscore):
            correct['sam'] += 1


print("Number of instances of current dataset counter:", counter)
print(correct)

for k,v in correct.items():
    print(v / 10000) 

