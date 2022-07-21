# BGNN-trait-segmentation
--------------------------

PyTorch implementation of Fish trait segmentation model.
This segmentation model is based on pretrained model using the [segementation models torch](https://github.com/qubvel/segmentation_models.pytorch)
Then the model is fine tune on fish image in order to identify (segment) the different traits.
Trait list : 
+ 'background': 
+ 'dorsal_fin'
+ 'adipos_fin'
+ 'caudal_fin'
+ 'anal_fin'
+ 'pelvic_fin'
+ 'pectoral_fin'
+ 'head'
+ 'eye'
+ 'caudal_fin_ray'
+ 'alt_fin_ray'
+ 'trunk'

**Structure of the repo**

## 1- Neural network for segmentation approach

## 2- Training

## 3- Training set annotation

## 4- Prediction

## 5- Input, output

## 6- Rescaling

## 7- Implicit/explicit code


### Requirements:
------------------
- Python (>=3.6)
- PyTorch
- **segmentation_models_pytorch**
- **albumentations**
- Numpy
- CV2
- Matplotlib
- Pillow
- Seaborn
- Pylab
- Pandas

### Source File
----------------
Run the **single_model_training.ipynb** notebook for training.

