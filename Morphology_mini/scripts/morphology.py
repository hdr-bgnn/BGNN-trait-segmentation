#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:02:59 2022

@author: thibault
"""

import os
import sys
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops, regionprops_table
from math import sqrt
import json


trait_list = ["background", "dorsal_fin", "adipos_fin", "caudal_fin", "anal_fin", "pelvic_fin", "pectoral_fin",
              "head", "eye", "caudal_fin_ray", "alt_fin_ray", "alt_fine_spine", "trunk"]

color_list = [np.array([0, 0, 0]), np.array([254, 0, 0]),
              np.array([0, 254, 0]), np.array([0, 0, 254]),
              np.array([254, 254, 0]), np.array([0, 254, 254]),
              np.array([254, 0, 254]), np.array([254, 254, 254]),
              np.array([0, 254, 102]), np.array([254, 102, 102]),
              np.array([254, 102, 204]), np.array([254, 204, 102]),
              np.array([0, 124, 124])]

# Create dictionnaries to go from channel to color or from trait to color
channel_color_dict = {i: j for i, j in enumerate(color_list)}
trait_color_dict = dict(zip(trait_list, color_list))


def import_segmented_image(image_path):
    '''
    import the image from "image_path" and convert to np.array astype uint8 (0-255)

    '''
    img = Image.open(image_path)
    img_arr = np.array(img, dtype=np.uint8)

    return img_arr


def get_one_trait_mask(img, trait_color_dict, trait_key):
    '''
    Create a mask for a trait define by "trait_key" using "img" (the image array)
    and trait_color_dict (the trait to color dictionnary) 

    '''
    color_array = trait_color_dict[trait_key]
    trait_mask = (img[:, :, 0] == color_array[0]) & (
        img[:, :, 1] == color_array[1]) & (img[:, :, 2] == color_array[2])

    return trait_mask


def get_channels_mask(img, trait_color_dict):
    ''' Convert the png image (numpy.ndarray, np.uint8)  (320, 800, 3)
    to a mask_channel (320, 800, 12) Binary map

    input
    output
    img shape -> (320, 800, 3) if normal=True
    else: mask shape -> (12, 320, 800)
        we want to output a PIL image with rgb color value
    '''

    mask = {}
    for color, trait in enumerate(trait_color_dict):

        mask[trait] = get_one_trait_mask(
            img, trait_color_dict, trait).astype("uint8")

    return mask

def get_scale(metadata_file):

    f = open(metadata_file)
    data = json.load(f)
    first_value = list(data.values())[0]

    if first_value['has_ruler']==True:

        scale = round(first_value['scale'],3)
    else: scale =[None]
    
    return scale


def get_morphology_one_trait(trait_key, mask, parameter=None):

    trait_mask = mask[trait_key]
    total_area = sum(sum(trait_mask))
    label_trait = label(trait_mask)
    regions_trait = regionprops(label_trait)

    result={"area":[], "percent" : [],"centroid":[], "bbox":[]}
    # iterate throught the region sorted by area size
    for region in sorted(regions_trait, key=lambda r: r.area, reverse=True):

        # choose what you want to see 
        result["area"].append(region.area)
        result["percent"].append(region.area/total_area)
        result["centroid"].append(region.centroid)
        result["bbox"].append(region.bbox)
    
    return result, regions_trait


def compare_head_eye(result_head, head, result_eye, eye, metadata_file,  name=None):

    # Checked if there is a major big blob
    if result_head["percent"][0] > 0.85 and result_eye["percent"][0] > 0.85:

        head = head[0]
        eye = eye[0]
        ratio_eye_head = eye.area/(head.area + eye.area)

        coord_head = head.centroid
        coord_eye = eye.centroid

        distance_eye_snout =  abs(head.bbox[1]-eye.bbox[1])
        scale = get_scale(metadata_file)
        
    return {name:{"eye_head_ratio" : round(ratio_eye_head,3),
                "snout_eye_distance": round(distance_eye_snout,3), "scale":scale}}

def main(image_path, metadata_file, output_json, name):


    img = import_segmented_image(image_path)

    mask = get_channels_mask(img, trait_color_dict)

    # Create your own function
    res_head, head = get_morphology_one_trait("head", mask, parameter=None)
    res_eye, eye = get_morphology_one_trait("eye", mask, parameter=None)

    result = compare_head_eye(res_head, head, res_eye, eye, metadata_file, name)

    with open(output_json, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':

    input_file = sys.argv[1]
    metadata_file = sys.argv[2]
    output_json = sys.argv[3]    
    name = sys.argv[4]
    main(input_file, metadata_file, output_json, name)
