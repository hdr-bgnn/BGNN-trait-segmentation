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

from scripts.preprocessing import Dataset
from scripts.helper import plot_double_image, plot_triple_image, get_color_img, get_concat_h
from scripts.augmentation import get_training_augmentation, get_validation_augmentation, to_tensor, get_preprocessing, get_preprocessing_unlabeled
from scripts.helper import get_midpoint, get_row_col_mat

# DEVICE = ('cuda:2' if torch.cuda.is_available() else 'cpu')

def dorsal_to(pred_mask, sub_mask, obj_mask, row_mat): 
    max_sub = np.max(row_mat * sub_mask)
    max_obj = np.max(row_mat * obj_mask)

    return float(max(0, max_sub - max_obj)>0)


def explicit_dorsal_to_batch(labelmaps, tensor=True):
    batch_size = labelmaps.shape[0]
    if tensor:
        y = labelmaps.detach().cpu().numpy()
    else:
        y = labelmaps
    penalty = 0
    for bs in range(batch_size):
        dorsal = y[bs, 0, :, :]
        adipose = y[bs, 1, :, :]
        caudal = y[bs, 2, :, :]
        anal = y[bs, 3, :, :]
        pelvic = y[bs, 4, :, :]
        pectoral = y[bs, 5, :, :]
        head = y[bs, 6, :, :]
        trunk = y[bs, 11, :, :]
        
        if dorsal.sum() == 0 or caudal.sum() == 0 or head.sum() == 0 or trunk.sum() == 0:
            continue
        dorsal_mid = get_midpoint(dorsal)
        caudal_mid = get_midpoint(caudal)
        head_mid = get_midpoint(head)
        trunk_mid = get_midpoint(trunk)
        row_mat, col_mat = get_row_col_mat(dorsal_mid, caudal_mid, head_mid, trunk_mid)
        
        if dorsal.sum() != 0 and pectoral.sum() != 0:
            penalty += dorsal_to(y[bs, :, :, :], dorsal, pectoral, row_mat)
            
        if dorsal.sum() != 0 and pelvic.sum() != 0:
            penalty += dorsal_to(y[bs, :, :, :], dorsal, pelvic, row_mat)
            
        if dorsal.sum() != 0 and anal.sum() != 0:
            penalty += dorsal_to(y[bs, :, :, :], dorsal, anal, row_mat)
            
    return penalty

def explicit_dorsal_to_nobatch(labelmap, tensor=True):
    if tensor:
        y = labelmap.detach().cpu().numpy()
    else:
        y = labelmap
        
    penalty = 0
    
    dorsal = y[0, :, :]
    adipose = y[1, :, :]
    caudal = y[2, :, :]
    anal = y[3, :, :]
    pelvic = y[4, :, :]
    pectoral = y[5, :, :]
    head = y[6, :, :]
    trunk = y[11, :, :]

    if dorsal.sum() == 0 or caudal.sum() == 0 or head.sum() == 0 or trunk.sum() == 0:
        return 0
    dorsal_mid = get_midpoint(dorsal)
    caudal_mid = get_midpoint(caudal)
    head_mid = get_midpoint(head)
    trunk_mid = get_midpoint(trunk)
    row_mat, col_mat = get_row_col_mat(dorsal_mid, caudal_mid, head_mid, trunk_mid)

    if dorsal.sum() != 0 and pectoral.sum() != 0:
        penalty += dorsal_to(y, dorsal, pectoral, row_mat)

    if dorsal.sum() != 0 and pelvic.sum() != 0:
        penalty += dorsal_to(y, dorsal, pelvic, row_mat)

    if dorsal.sum() != 0 and anal.sum() != 0:
        penalty += dorsal_to(y, dorsal, anal, row_mat)
        
    return penalty


def rn_dorsal_to_batch(labelmaps, rn_model, tensor=True, device='cpu'):
    if tensor == False:
        prediction = torch.tensor(labelmaps).float()
    else:
        prediction = labelmaps
    D = prediction[:, 0, :, :]
    Pel = prediction[:, 4, :, :]
    Pec = prediction[:, 5, :, :]
    An = prediction[:,3, :, :]
    H = prediction[:, 6, :, :]
    T = prediction[:, 2, :, :]
    
    penalty = 0
    
    # Dorsal-Pectoral
    if D.sum() != 0 and Pec.sum() != 0:
        rn_inp = torch.stack((H, T, D, Pec), dim=1).to(device)
        y_pred = rn_model(rn_inp)
        penalty += (y_pred<0.5).float().sum().item()
        
    # Dorsal-Pelvic
    if D.sum() != 0 and Pel.sum() != 0:
        rn_inp = torch.stack((H, T, D, Pel), dim=1).to(device)
        y_pred = rn_model(rn_inp)
        penalty += (y_pred<0.5).float().sum().item()
        
    # Dorsal-Anal
    if D.sum() != 0 and An.sum() != 0:
        rn_inp = torch.stack((H, T, D, An), dim=1).to(device)
        y_pred = rn_model(rn_inp)
        penalty += (y_pred<0.5).float().sum().item()
    return penalty


def rn_dorsal_to_nobatch(labelmaps, rn_model, tensor=True, device='cpu'):
    if tensor == False:
        prediction = torch.tensor(labelmaps).float()
    else:
        prediction = labelmaps
    labelmaps = torch.zeros(1, prediction.shape[0], prediction.shape[1], prediction.shape[2])
    labelmaps[0, :, :, :] = prediction
    
    return rn_dorsal_to_batch(labelmaps, rn_model, tensor=True, device=device)