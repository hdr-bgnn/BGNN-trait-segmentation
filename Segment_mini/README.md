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
```conda create -n seg_min -f env_segment_mini.yml```

# 3- Download model weights
    There are 2 models used in this code:
    
## 1- Model weights trained by Maruf using annotated fish:
    you can download the model using
```
cd BGNN-trait-segmentation/Segment_mini_dev/scripts
chmod +x load_model.sh
./load_model.sh
```
    If this doesn't work, do it manually. Copy/past in your browser  https://drive.google.com/uc?id=1HBSGXbWw5Vorj82buF-gCi6S2DpF4mFL
    Copy the file "Trained_model_SM.pth" to BGNN-trait-segmentation/Segment_mini_dev/scripts/saved_models
    
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
    
# 4- Test and Usage

## Clone the repository on you system and checkout the segment_min branch

```
git clone 
git checkout segment_mini
```


## activate your environment
```conda activate seg_min```

## Test 

```
cd BGNN-trait-segmentation/Segment_mini_dev/
python scripts/segmentation_main.py image_test/INHS_FISH_79829_cropped.jpg seg.png
```

## Usage 

Same as for Test

```
cd BGNN-trait-segmentation/Segment_mini_dev/
python scripts/segmentation_main.py image/location/fish.jpg result/location/res.png
```

## Export an environment of your own 
For instance if you want to add some functionalities that required new packages.

```conda env export > environment.yml```


# 5- Create the singularity image

This section describe steps use to create 2 types of containers.

1. One that provides the environment to run segmentation_main.py from outisde the container using singularity. This is a good way to develop the code if you don't want to setup a conda enironment.
2. One to run trait segmentation (segmentation_main.py) inside the container. In this case everything is self-contained. Just run the container by itself.


The image is available for download at 

##  1. Create container for the environment only
With this approach you have to execute and a script which is located outside the container.

### Create the Dockerfile that contained the environment

I provide a template based on anibali/pytorch:1.10.0-nocuda
for which I change the miniconda py39 to py38

1. Build the docker image containing dependency to run segment_mini
```
cd Segment_mini_dev
docker build -t smp_env:v2 -f Docker_SMP_env_v2/Dockerfile_smp_env_v2
```
2. Convert the docker image in singularity (from the local version (to see the list command line "docker images")

```sudo singularity build smp_env_v2.sif docker-daemon:smp_env:v2```

3. Test it
```
cd Segment_mini_dev
singularity shell smp_env_v2.sif
```

In the shell of the container 
```
-> python segmentation_main.py image_test/INHS_FISH_79829_cropped.jpg result_segment.png
-> exit # result_segment.png should be on your local host in pwd 
```

Back the host
```
ls # should display result_segment.png
```

##  2. Create a container that runs segmentation from inside
In the previous section we have create a singularity image smp_env_v2.sif provide only the environment for segmentation_models_pytorch.
It doesn't contain any code to run segmentation_main.py which is located outside the container. This new container we will create, we will run every everything from inside and is completely protable.


### Create the container that contains everything

```
cd Segment_mini_dev/scripts
chmod +x create_simg.sh
./create_simg.sh
```
It created a singularity image file (.sif) segment_mini.sif

### Test and Usage

```
singularity exec segment_min.sif segmentation_main.py image_test/INHs...jpg result_segment.png
```
