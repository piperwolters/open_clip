import json
import numpy as np
import os
import shutil

annotation_fname = '/home/ubuntu/satlas_event_task/superres_naip_human_feedback_data/batch02/annotations.json'
src_dir = '/multisat/datasets/superres/naip_human_feedback_eval/'
dst_dir = '/multisat/datasets/superres/naip_human_feedback_finetune/'

with open(annotation_fname, 'r') as f:
    annotations = json.load(f)

by_example = {}
for ann in annotations:
    parts = ann['image_url'].split(',')
    tile_name = parts[0].split('/')[-1]
    left_name = parts[1]
    right_name = parts[2]
    k = (tile_name, left_name, right_name)
    if k not in by_example:
        by_example[k] = []
    by_example[k].append(ann['answer'])

for (tile_name, left_name, right_name), answers in by_example.items():
    answer = np.mean(answers)
    if answer < 0.75:
        good_name = left_name
        bad_name = right_name
    elif answer > 1.25:
        good_name = right_name
        bad_name = left_name
    else:
        continue

    cur_src_dir = os.path.join(src_dir, tile_name)
    cur_dst_dir = os.path.join(dst_dir, tile_name)
    os.makedirs(cur_dst_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(cur_src_dir, 'naip.png'),
        os.path.join(cur_dst_dir, 'naip.png'),
    )
    shutil.copyfile(
        os.path.join(cur_src_dir, '{}.png'.format(good_name)),
        os.path.join(cur_dst_dir, 'good.png'),
    )
    shutil.copyfile(
        os.path.join(cur_src_dir, '{}.png'.format(bad_name)),
        os.path.join(cur_dst_dir, 'bad.png'),
    )