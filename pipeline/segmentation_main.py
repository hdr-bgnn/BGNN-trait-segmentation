#!/bin/local/python

#import os
import sys
import numpy as np
#import pandas as pd
#import cv2
from PIL import Image

#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset as BaseDataset

import torch
#import segmentation_models_pytorch as smp
#import albumentations as albu

#import seaborn as sns
#import pylab as py
#import pandas as pd
#import torch
from torchvision import transforms

import scripts.helper_T as sh

DEVICE = ('cuda:4' if torch.cuda.is_available() else 'cpu')
model, preprocessing_fn, CLASSES = sh.load_pretrained_model()
map_location=torch.device('cpu')
model = torch.load('saved_models/Trained_model_SM.pth', map_location=map_location)


def Import_image (file_path):
    img = Image.open(file_path)
    return img

def Resize_img(img, r_width = 800, r_height = 320 ):
    '''
    Resize the image to the size r_width x r_height
    Default value 800x320

    input : File_path is the location of the image(.png or .jpg)
    ouput : image resized in the form of numpy array  '''

    r_img = transforms.Resize((r_height, r_width))(img)
    #r_img.save('transformed_images/'+image_name)
    r_img = np.array(r_img, dtype=np.float32)
    return r_img

def Preprocessing_single_image(img_PIL):
    '''
    From a image in PIL format, Resize the image to 800x320 in np.array, Normalize using the preprocessing_fn function
    and finally format the order of axis [ch, width, height]

    input : img_PIL image in PIL format
    output : image_array, numpy array with type float32 3x800x320
    '''
    img_array = Resize_img(img_PIL)
    image_array = preprocessing_fn(img_array).astype('float32')
    image_array = np.moveaxis(image_array, -1, 0)
    return image_array

# functions to perform prediction

def Traits_prediction_one_image(image_array):
    '''
    Predict the traits from a image array which should be resized and
    normalized by Preprocessing single_image()

    input: image_array image in numpy array type float32
    output: pred_mask, prediction mask from model as numpy 12x800x320
    '''

    # Create a 4 dimension tensor require fr the model [batch, width, height, channel]
    img_tensor = torch.from_numpy(image_array).to(DEVICE).unsqueeze(0)
    pred_mask = model.predict(img_tensor)
    pred_mask = pred_mask.squeeze().cpu().numpy().round()

    return pred_mask

def main(image_path, output_path):
    '''
        input:
            image_path: location of the image.jpg
            output_path: location to save the segmentation.png
        ouput:
            None
            
            
    '''
    # Preprocess
    img = Import_image (image_path)
    image_preprocess = Preprocessing_single_image(img)

    # Prediction
    pred_mask = Traits_prediction_one_image(image_preprocess)    
    
    # Save
    sh.get_color_img(pred_mask, normal=False).save(output_path)
    
if __name__ == '__main__':
    
    main(sys.argv[1],sys.argv[2])