import json
from statistics import mean

model_names = ['srcnn','highresnet','sr3','sr3_cfg','esrgan_satlas','esrgan_satlas_chkpt5k',
                'esrgan_satlas_chkpt50k','esrgan_osm','esrgan_osm_chkpt5k','esrgan_osm_chkpt50k']


### First get list of binary decisions for mturk human evaluation ###

annots_file = open('/data/piperw/scripts/mturk_batch01_dict2.json')
annots = json.load(annots_file)

human_results = {}
# For each m1-m2 combination, we just want to know the average score of this combination and a binary decision.
for m1 in model_names:
    for m2 in model_names:
        if m1 == m2:
            continue
        human_results[m1 + '-' + m2] = []

print("Iterating through ", len(annots.items()), " datapoints.")
for idx,(chip, d) in enumerate(annots.items()):
    model1_name = d['model1']
    model2_name = d['model2']
    answers = d['answers']
    avg_answer = mean(answers)

    if avg_answer < 1:
        human_results[model1_name+'-'+model2_name].append(avg_answer)  # left better
    elif avg_answer > 1:
        human_results[model1_name+'-'+model2_name].append(avg_answer)  # right better
    else:
        human_results[model1_name+'-'+model2_name].append(avg_answer)  # same


### Then get same list of binary decisions for each metric ###

metric_names = ['psnr', 'ssim', 'lpips_alex', 'lpips_vgg', 'clip', 'naip_clip', 'siglip', 'dino', 'satlas_backbone', 'satlas_fpn']

f = open('compare_mturk_to_metrics.json')
metrics = json.load(f)

# Initialize dicts for each m1-m2 combination, for each of the metrics.
psnr, ssim, alex, vgg, clip, naip_clip, siglip, dino, satlas_backbone, satlas_fpn = {},{},{},{},{},{},{},{},{},{}
metric_dicts = [psnr, ssim, alex, vgg, clip, naip_clip, siglip, dino, satlas_backbone, satlas_fpn]
for m1 in model_names:
    for m2 in model_names:
        if m1 == m2:
            continue
        psnr[m1 + '-' + m2] = []
        ssim[m1 + '-' + m2] = []
        alex[m1 + '-' + m2] = []
        vgg[m1 + '-' + m2] = []
        clip[m1 + '-' + m2] = []
        naip_clip[m1 + '-' + m2] = []
        siglip[m1 + '-' + m2] = []
        dino[m1 + '-' + m2] = []
        satlas_backbone[m1 + '-' + m2] = []
        satlas_fpn[m1 + '-' + m2] = []

for k,v in metrics.items():

    # Skip keys with hr_x4, hr_x8, hr_x16....oops
    if 'hr_x' in k:
        continue

    models = list(metrics[k].keys())
    m1, m2 = models[0], models[1]
    dk = m1 + '-' + m2

    for i,metric in enumerate(metric_names):
        avg_m1 = mean(v[m1][metric])
        avg_m2 = mean(v[m2][metric])

        if avg_m1 > avg_m2:
            metric_dicts[i][dk].append(0)
        elif avg_m1 < avg_m2:
            metric_dicts[i][dk].append(2)
        else:
            metric_dicts[i][dk].append(1)


### Now compute the % agreement between mturk human evaluation and each metric ###
# Compare human_results to each of the metrics in metric_dicts.
agreement = {}
for m,metric_dict in enumerate(metric_dicts):

    agree, disagree = 0.0, 0.0
    for model_combo, winner in metric_dict.items():
        metric_guess = winner[0]
        human_guess = mean(human_results[model_combo])
        human_guess = 0 if human_guess < 1 else 2

        if human_guess == metric_guess:
            agree += 1
        else:
            disagree += 1
    
    print(metric_names[m], " agrees:", agree, " and disagrees:", disagree)
