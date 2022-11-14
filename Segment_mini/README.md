# Trait-segmentation-minimum
This is a branch from Marufv which is called Segment_mini_dev. It contain the minimum required to run the trait segmentation fish.
For the detail on training the model, the architecture and the training data looks in the main branch or contact Maruf v at () 

# 1-Introduction

# 2- Setup and Requirements

## Install conda if you don't have it

There are plenty of option, here is one
https://docs.conda.io/en/master/miniconda.html

Or manually
    + download in ~/
    https://repo.anaconda.com/miniconda/Miniconda-3.8.3-Linux-x86_64.sh
    + change the permission (make miniconda.sh executable)
    
```chmod +x ~/miniconda.sh ``` 
    + Execute .sh
    
``` ~/miniconda.sh -b -p ~/miniconda ```


## Create conda environment manually
```
conda create -n seg_mini python=3.8
pip install numpy pillow
pip install matplotlib
pip install segmentation_models_pytorch
```
## Create conda environment using .yml file 

```
git clone git@github.com:hdr-bgnn/BGNN-trait-segmentation.git
cd BGNN-trait-segmentation/Segment_mini/
conda create -n seg_min -f env_segment_mini.yml
```

# 3- Download model weights
There are 2 models used in this code:
    
## 1- Model weights trained by Maruf using annotated fish:
To Download the model you can use

```
cd BGNN-trait-segmentation/Segment_mini_dev/scripts
chmod +x load_model.sh
./load_model.sh
```
Which downlaod it from google dirve.
If this doesn't work, do it manually. Copy/past in your browser  https://drive.google.com/uc?id=1HBSGXbWw5Vorj82buF-gCi6S2DpF4mFL.
Copy the file "Trained_model_SM.pth" to BGNN-trait-segmentation/Segment_mini_dev/scripts/saved_models

Alternatively use [dva](https://github.com/Imageomics/dataverse-access) if you have access to the [Datacommun](https://datacommons.tdai.osu.edu/dataverse/fish-traits/).
This feature will be use by default when everything is publicly available

```
dva download doi:10.5072/FK2/SWV0YL /model --url https://datacommons.tdai.osu.edu/
```

## 2- Pretrained model weights:    
This model parameter will be downloaded automatically when you run the code for the first time.
You can use scripts/load_pretrained_model.py to do it manually.
The weights/model should be store ~/.cache/torch/hub/checkpoint/se_resnext50_32x4d-a260b3a4.pth
To check the location of the weights 
    
```
python
import torch
torch.hub.get_dir()
```
Those weights are included in the container

If it does work you can download manually using the [dva tool](https://github.com/Imageomics/dataverse-access). 
In this case the file is publicly available.

```
dva download doi:10.5072/FK2/CGWDW4 ~/.cache/torch/hub/checkpoint/ --url https://datacommons.tdai.osu.edu/
```
    
# 4- Test and Usage


## Activate your environment
```
conda activate seg_min
```


## Test/Usage

```
cd BGNN-trait-segmentation/Segment_mini_dev/
python scripts/segmentation_main.py image_test/INHS_142865_cropped.jpg  image_test/segmentation_142865.png
```

# 5- Container

## Build the docker image
DATAVERSE_API_TOKEN required.
Availible to member with access to the [DATACOMMUN](https://datacommons.tdai.osu.edu/dataverse/fish-traits/)

```
cd BGNN-trait-segmentation/Segment_mini/
export DATAVERSE_API_TOKEN=xxxxxxx
docker build -t bgnn-trait-segmentation:dev --arg-buid DATAVERSE_API_TOKEN .
```

A version of the image is availible for test [here](https://github.com/hdr-bgnn/BGNN-trait-segmentation/pkgs/container/bgnn-trait-segmentation)

```
docker pull ghcr.io/hdr-bgnn/bgnn-trait-segmentation:0.0.6
```

```
singularity pull docker://ghcr.io/hdr-bgnn/bgnn-trait-segmentation:0.0.6
```


## Test it

Check usage:
```
singularity run bgnn-trait-segmentation:0.0.6.sif
```

Usage example:

```
cd BGNN-trait-segmentation/Segment_mini/
singularity exec bgnn-trait-segmentation:0.0.6.sif segmentation_main_rescale_origin.py image_test/INHS_FISH_79829_cropped.jpg seg.png
```




