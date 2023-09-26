#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
sys.path.append(current_path)

from torch.utils.data import Dataset
from saicinpainting.evaluation.data import scale_image, pad_img_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

class InpaintingDataset(Dataset):
    def __init__(self, img, mask, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.mask = mask
        self.img = img
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return 1

    def __getitem__(self, i):
        self.img = self.img.transpose((2, 0, 1))
        result = dict(image=self.img, mask=self.mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

def inpaint(img, mask):    
    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    indir = current_path + '/data'
    model_path = current_path + '/big-lama'
    checkpoint = 'best.ckpt'
    dataset_img_suffix = '.png'
    dataset_pad_out_to_modulo = 8
    out_key = 'inpainted'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_path, 'models', checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    if not indir.endswith('/'):
        indir += '/'

    dataset = InpaintingDataset(img, mask, img_suffix=dataset_img_suffix, pad_out_to_modulo=dataset_pad_out_to_modulo)
    for img_i in tqdm.trange(len(dataset)):
        batch = default_collate([dataset[img_i]])
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)                    
            cur_res = batch[out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    
    return cur_res
