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

def get_midpoint(mask):
    if mask.sum() < 10:
        return -1, -1
    else:
        r, c = np.where(mask==1)
        max_r, min_r = np.max(r), np.min(r)
        max_c, min_c = np.max(c), np.min(c)

        m_r, m_c = round((max_r + min_r)/2), round((max_c + min_c)/2)
        
        return m_r, m_c
    

def rotated_points(theta, points):
    A = np.array([[np.cos(theta), -1*np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.matmul(A, points.T).T

def creat_row_col_matrix(theta=0, row_num=320, col_num=800, rev=False):
    row_matrix = np.zeros((row_num, col_num))
    row_matrix_rev = np.zeros((row_num, col_num))
    col_matrix = np.zeros((row_num, col_num))
    
    points = []
    for i in range(row_num):
        for j in range(col_num):
            points.append([i, j])
    points = np.array(points)
    
    
    rotated_pt = rotated_points(theta, points)
    
    for i, pt in enumerate(points):
        row_matrix[pt[0], pt[1]] = rotated_pt[i, 0]
        col_matrix[pt[0], pt[1]] = rotated_pt[i, 1]
        
    if row_matrix.min() < 0:
        row_matrix -= row_matrix.min()
    
    if col_matrix.min() < 0:
        col_matrix -= col_matrix.min()
    
    need = False
    if np.min(row_matrix[0, :]) > np.min(row_matrix[-1, :]):
        need = True
#     print(rev)
#     print(need)
    if rev == False and need == False:
        return row_matrix, col_matrix
        
    for i, row in enumerate(row_matrix):
        row_matrix_rev[(row_num - i - 1), :] = row
            
    return row_matrix_rev, col_matrix

def get_theta(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def dorsal_to(pred_mask, sub_mask, obj_mask, row_mat):
    
    max_sub = torch.max(row_mat * sub_mask)
    max_obj = torch.max(row_mat * obj_mask)
    
    loss = torch.nn.functional.relu(max_sub - max_obj)

    loss = (1-torch.exp(-1*loss))

    return loss

def anterior_to_normal(pred_mask, sub_mask, obj_mask, col_mat):
    max_sub = torch.max(col_mat * sub_mask)
    max_obj = torch.max(col_mat * obj_mask)
    
    loss = torch.nn.functional.relu(max_sub - max_obj)
    gradient = torch.autograd.grad(loss, pred_mask)[0]
    return loss, gradient

def anterior_to(pred_mask, sub_mask, obj_mask, col_mat):
    max_sub = torch.max(col_mat * sub_mask)
    max_obj = torch.max(col_mat * obj_mask)
    
    loss = torch.nn.functional.relu(max_sub - max_obj)
    loss = (1-torch.exp(-1*loss))
#     gradient = torch.autograd.grad(loss, pred_mask)[0]
    return loss

def anterior_to_rev(pred_mask, sub_mask, obj_mask, col_mat):
    max_sub = torch.max(col_mat * sub_mask)
    max_obj = torch.max(col_mat * obj_mask)
    
    loss = torch.nn.functional.relu(- max_sub + max_obj)
    gradient = torch.autograd.grad(loss, pred_mask)[0]
    return loss, gradient

def print_row_col(col_mat):
    list_all = []
    for i in col_mat:
        list_ = []
        for j in i:
            list_.append(round(j))
        list_all.append(list_)
    print(np.array(list_all))
    
def get_row_col_mat(dorsal_mid, caudal_mid, head_mid, trunk_mid):
    # get the rotation direction w.r.t. a base 
    base_v = np.array([1, 0])
    rev = dorsal_mid[0] > trunk_mid[0]
    theta = get_theta(base_v, np.array([caudal_mid[1]-head_mid[1], caudal_mid[0]-head_mid[0]]))
    row_mat, col_mat = creat_row_col_matrix(theta=theta, row_num=320, col_num=800, rev=rev)
    return row_mat, col_mat
    


