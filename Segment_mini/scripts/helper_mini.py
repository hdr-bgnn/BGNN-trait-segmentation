#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
from M. Maruf modified by Thibault Tabarin
"""

#import os
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

def get_color_img(mask, normal=True):
    ''' mask shape -> (320, 800, 12) if normal=True
    else: mask shape -> (12, 320, 800)
        we want to output a PIL image with rgb color value
    '''
    colors = {}                            #   Name         Label
    #                                      # ----------------------
    colors[0] = np.array([0, 0, 0])        # Background      0
    colors[1] = np.array([254, 0, 0])      # Dorsal Fin      1
    colors[2] = np.array([0, 254, 0])      # Adipos Fin      2
    colors[3] = np.array([0, 0, 254])      # Caudal Fin      3
    colors[4] = np.array([254, 254, 0])    # Anal Fin        4
    colors[5] = np.array([0, 254, 254])    # Pelvic Fin      5
    colors[6] = np.array([254, 0, 254])    # Pectoral Fin    6
    colors[7] = np.array([254, 254, 254])  # Head            7
    colors[8] = np.array([0, 254, 102])    # Eye             8
    colors[9] = np.array([254, 102, 102])  # Caudal Fin Ray  9
    colors[10] = np.array([254, 102, 204]) # Alt Fin Ray     10
    colors[11] = np.array([254, 204, 102]) # Alt Fin Spine   11
    colors[12] = np.array([0, 124, 124])   # Trunk           12

    if normal == True:
        color_data = np.zeros((mask.shape[0], \
                               mask.shape[1], 3)).astype(np.uint8)

        for ch in range(0, mask.shape[2]):
            if mask[:, :, ch].sum() == 0:
                continue
            else:
                row_, col_ = np.where(mask[:, :, ch] == 1)
                color_data[row_, col_] = colors[ch+1]

        img_ = Image.fromarray(color_data)


        return img_

    else:
        color_data = np.zeros((mask.shape[1], \
                               mask.shape[2], 3)).astype(np.uint8)

        for ch in range(0, mask.shape[0]):
            if mask[ch, :, :].sum() == 0:
                continue
            else:
                row_, col_ = np.where(mask[ch, :, :] == 1)
                color_data[row_, col_] = colors[ch+1]
        img_ = Image.fromarray(color_data)


        return img_

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def load_pretrained_model():
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['dorsal', 'adipose', 'caudal', 'anal',
               'pelvic', 'pectoral', 'head', 'eye',
               'caudal-ray', 'alt-ray', 'alt-spine', 'trunk']
    ACTIVATION = 'sigmoid'

    # segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name = ENCODER,
        encoder_weights = ENCODER_WEIGHTS,
        classes = len(CLASSES),
        activation = ACTIVATION,
    )
    # preprocessing function for this encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

    return model, preprocessing_fn, CLASSES
