import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import torch
import segmentation_models_pytorch as smp
import albumentations as albu

import seaborn as sns
import pylab as py
import pandas as pd


class Dataset(BaseDataset):
    ''' Works for our fish dataset.
    This module will read image from the location/directory you have provided.
    This module can apply any default augmentation or can apply preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    '''
    
    # The label set this dataset has:    
    CLASSES = ['background', 'dorsal', 'adipose', 'caudal', 'anal', 
               'pelvic', 'pectoral', 'head', 'eye', 
               'caudal-ray', 'alt-ray', 'alt-spine', 'trunk']
    
    def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None, preprocessing=None,):
        
        # list all fils in images_dir folder
        self.ids = os.listdir(images_dir)
        
        
        # remove all non-image/non-mask files
        # self.images_fps contains image list
        # self.masks_fps contains mask list
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.images_fps = [items for items in self.images_fps if items[-4:]=='.png']
        
        if masks_dir is not None:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
            self.masks_fps = [items for items in self.masks_fps if items[-4:]=='.png']
        
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        # initialize augmentation and preprocessing 
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.masks_dir = masks_dir
        
    def __getitem__(self, i):
        
        # this module will run for each image with index i
        
        # read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read the mask
        if self.masks_dir is not None:
            mask = cv2.imread(self.masks_fps[i], 0)
        
        
            # separate channel for different masks -> [:, :, channel]
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            if self.masks_dir is not None:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.augmentation(image=image)
                image = sample['image']
                mask = None
        
        # apply preprocessing
        if self.preprocessing:
            if self.masks_dir is not None:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']
                mask = None
            
        return image, mask
    
    
    def __len__(self):
        return len(self.images_fps)  