from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def plot_image(data):
    ''' plots image from numpy array
    '''
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)
    plt.show()
    
def plot_pil_image(image):
    ''' plots image from pil image
    '''
    data = np.asarray(image)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)
    plt.show()
    
    
def plot_colormap(labelmap_data):
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
    
    color_data = np.zeros((labelmap_data.shape[0], labelmap_data.shape[1], 3)).astype(np.uint8)
    
    for ch in range(1, data.shape[2]):
        if labelmap_data[:, :, ch].sum() == 0:
            continue
        else:
            row_, col_ = np.where(labelmap_data[:, :, ch] == 1)
            color_data[row_, col_] = colors[ch]
            
    img_ = Image.fromarray(color_data)
    return img_

def plot_colormap_pil_image(labelmap_img):
    
    labelmap_data = np.asarray(labelmap_img)
    
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
    
    color_data = np.zeros((labelmap_data.shape[0], labelmap_data.shape[1], 3)).astype(np.uint8)
    
    for ch in range(1, 13):
        row_, col_ = np.where(labelmap_data[:, :] == ch)
        color_data[row_, col_] = colors[ch]
            
    img_ = Image.fromarray(color_data)
    return img_
    
    
    