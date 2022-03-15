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
    + change the permission (make miniconda.sh executable
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

# 3- Test and Usage

## activate your environment
```conda activate seg_min```

## Test 

```python segmentation_main.py INHS_FISH_79829_cropped.jpg seg.png```

## Usage 

Same as for Test


## Export an environment of your own 
For instance if you want to add some functionalities that required new package.
```conda env export > environment.yml```


# 4- Create the singularity image

This section describe steps use to create 2 types of containers : 
    1. One that provides the environment to run segmentation_main.py from outisde the container using singularity. This is a good way to develop the code if you don't want to setup a conda enironment.
    2. One to run trait segmentation (segmentation_main.py) inside the container. In this case everything is self-contained. Just run the container by itself.


The image is available for download at 

##  1. For detail on how to create continous to read more

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

Singularity image smp_env_v2.sif provide only the environment for segmentation_models_pytorch.
It doesn't contain any code to run segmentation_main.py which located outside the container

4- Create the container that contains everything

cd Segment_mini_dev
chmod +x create_simg.sh
./create_simg.sh
# it created a file segment_mini.sif

