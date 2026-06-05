import os
import cv2
import torch
import tronco
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def show_masks_on_image(masks, ax):
    # test whether masks were found
    if len(masks) == 0: return
    # small spheres are shown on top of big ones
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    img_shape = (sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4)
    overlay_img = np.ones(img_shape)
    overlay_img[:,:,3] = 0
    for mask_data in sorted_masks:
        binary_mask = mask_data['segmentation']
        random_color = np.concatenate([np.random.random(3), [0.45]]) 
        overlay_img[binary_mask] = random_color
    ax.imshow(overlay_img)

# initialise date and time
init = datetime.datetime.now()
date = init.date()
time = init.time().replace(microsecond=0)
# initialise logger
logger = tronco.tronco()
# initialise argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--example', action='store_true', help='Run example.')
parser.add_argument('--filename', type=str, help='Image name (full relative path, e.g., `raw/image-name.jpg`).')
parser.add_argument('--ratio', type=float, help='µm-to-pixel ratio.')
args = parser.parse_args()
# test args
isExample = False
if args.example:
    isExample = True
    image_path = 'raw/Example.jpg'
else:
    if args.filename:
        image_path = args.filename
    else:
        raise ValueError('Missing arguments! Specify the path to the image to be segmented.')

# (optional) µm-to-pixel ratio
ratio = 1.
isRatio = False
if args.ratio:
    ratio = args.ratio
    isRatio = True

# load the proper device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
logger.info(f'Running on {device}.')

# initialise automatic mask generator (paper: https://arxiv.org/abs/2408.00714)
sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=15,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
)

# load image
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image_bgr is None: raise FileNotFoundError(f'Image not found: {image_path}.')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
logger.info('Loaded image.')

# generate masks
logger.info('Generating masks.')
masks = mask_generator.generate(image_rgb)
delta_time = str(datetime.datetime.now() - init).split('.')[0]
logger.info(f'Evaluation time: {delta_time}.')

# define output name
out_name = f'Segmented-{date}-{time}'
if isExample: out_name = 'Segmented-Example'
if isRatio: out_name += '-µm' 

# creating pandas dataframe
df = pd.DataFrame([{'area': mask_data['area']*ratio**2,                                             # area
                    'radius': np.sqrt(mask_data['area']/np.pi)*ratio,                               # assuming circular object
                    'min-radius': np.min(mask_data['bbox'][2:])*ratio/2.,                           # minimum of width and height divided by 2
                    'max-radius': np.max(mask_data['bbox'][2:])*ratio/2.,                           # maximum of width and height divided by 2
                    'diag-radius': np.sqrt(np.sum(np.array(mask_data['bbox'][2:])**2))*ratio/2.,    # diagonal of bbox divided by 2
                    'bbox': mask_data['bbox']} for mask_data in masks],                             # info about the bbox in pixels
                    )
if not os.path.isdir('data'): os.mkdir('data')
df.to_csv(f'data/{out_name}.csv', index=False)
logger.info(f'Outputted data to data/{out_name}.csv')

# output segmented figure
fig, ax = plt.subplots(1, 1, figsize=(12, 12), layout='constrained')
ax.imshow(image_rgb)
show_masks_on_image(masks, ax)
ax.set_axis_off()
if not os.path.isdir('processed'): os.mkdir('processed')
fig.savefig(f'processed/{out_name}.jpg')
logger.info(f'Outputted segmented image to processed/{out_name}.jpg')