#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:56:12 2022

@author: thibault
"""

#import os
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from PIL import Image

#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset as BaseDataset

#import torch
import segmentation_models_pytorch as smp
#import albumentations as albu

#import seaborn as sns
#import pandas as pd


# class Dataset(BaseDataset):
#     ''' Works for our fish dataset.
#     This module will read image from the location/directory you have provided.
#     This module can apply any default augmentation or can apply preprocessing transformations.
    
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline 
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing 
#             (e.g. normalization, shape manipulation, etc.)
#     '''
    
#     # The label set this dataset has:    
#     CLASSES = ['background', 'dorsal', 'adipose', 'caudal', 'anal', 
#                'pelvic', 'pectoral', 'head', 'eye', 
#                'caudal-ray', 'alt-ray', 'alt-spine', 'trunk']
    
#     def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None, preprocessing=None,):
        
#         # list all fils in images_dir folder
#         self.ids = os.listdir(images_dir)
        
        
#         # remove all non-image/non-mask files
#         # self.images_fps contains image list
#         # self.masks_fps contains mask list
        
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.images_fps = [items for items in self.images_fps if items[-4:]=='.png']
        
#         if masks_dir is not None:
#             self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#             self.masks_fps = [items for items in self.masks_fps if items[-4:]=='.png']
        
        
#         # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
#         # initialize augmentation and preprocessing 
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#         self.masks_dir = masks_dir
        
#     def __getitem__(self, i):
        
#         # this module will run for each image with index i
        
#         # read the image
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # read the mask
#         if self.masks_dir is not None:
#             mask = cv2.imread(self.masks_fps[i], 0)
        
        
#             # separate channel for different masks -> [:, :, channel]
#             masks = [(mask == v) for v in self.class_values]
#             mask = np.stack(masks, axis=-1).astype('float')
#         else:
#             mask = None
        
#         # apply augmentations
#         if self.augmentation:
#             if self.masks_dir is not None:
#                 sample = self.augmentation(image=image, mask=mask)
#                 image, mask = sample['image'], sample['mask']
#             else:
#                 sample = self.augmentation(image=image)
#                 image = sample['image']
#                 mask = None
        
#         # apply preprocessing
#         if self.preprocessing:
#             if self.masks_dir is not None:
#                 sample = self.preprocessing(image=image, mask=mask)
#                 image, mask = sample['image'], sample['mask']
#             else:
#                 sample = self.preprocessing(image=image)
#                 image = sample['image']
#                 mask = None
            
#         return image, mask
    
    
#     def __len__(self):
#         return len(self.images_fps)  
    
   
    
    
    
def plot_double_image(image_1, image_2, title_1=None, title_2=None):
    ''' This method plot two images side by side
    '''
    
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    if title_1 != None:
        plt.title(title_1)
    plt.imshow(image_1)
    
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    if title_2 != None:
        plt.title(title_2)
    plt.imshow(image_2)
    plt.show()
    
def plot_triple_image(image_1, image_2, image_3, 
                      title_1=None, title_2=None, title_3=None):
    ''' This method plot two images side by side
    '''
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    if title_1 != None:
        plt.title(title_1)
    plt.imshow(image_1)
    
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    if title_2 != None:
        plt.title(title_2)
    plt.imshow(image_2)
    plt.show()
    
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    if title_3 != None:
        plt.title(title_3)
    plt.imshow(image_3)
    plt.show()
    
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



'''

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 800)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_preprocessing_unlabeled(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

'''

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
    