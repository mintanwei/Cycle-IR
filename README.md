# Cycle-IR: Deep Cyclic Image Retargeting
# 1. Overview

## Paper
This is a Tensorflow implementation of Cycle-IR approach for content-aware image retargeting. The munuscript is available at: https://ieeexplore.ieee.org/document/8943352.

![](https://github.com/mintanwei/Cycle-IR/blob/master/AdjustmentOfAspectRatio.png)

## Result
The retargeting results  of our approach on RetargetMe dataset are available at the folder "Our Cycle-IR result".

## Citation
@ARTICLE{CycleIR_TMM2019,
author={W. {Tan} and B. {Yan} and C. {Lin} and X. {Niu}},
journal={IEEE Transactions on Multimedia (TMM'2019)},
title={Cycle-IR: Deep Cyclic Image Retargeting},
year={2019},
volume={},
number={},
pages={1-1},
doi={10.1109/TMM.2019.2959925},
ISSN={1941-0077},
month={},}

This project includes the source code of TensorFlow implementation for our munuscript of "Cycle-IR: Deep Cyclic Image Retargeting". We demonstrate that image retargeting problem can be solved by using a promising way of unsupervised deep learning.

# 2. System requirements
  
  ## 2.1 Hardware Requirements
	The package requires only a standard computer with GPU and enough RAM to support the operations defined by a user. 
    For optimal performance, we recommend a computer with the following specs:
    RAM: 32+ GB
    CPU: 8+ cores, 3.6+ GHz/core
    GPU：GeForce RTX 1080 Ti GPU
  
  ## 2.2 Software Requirements
    numpy 1.15.4
    tensorflow 1.6.0
    scipy 1.1.0
    scikit-learn 0.20.2
    scikit-image 0.14.1
    opencv-python 3.3.0.10
    matplotlib 3.0.2
    pillow 5.3.0
	 
# 3. Installation Guide
  A working version of CUDA, python and tensorflow. This should be easy and simple installation. 
  CUDA(https://developer.nvidia.com/cuda-downloads)
  tensorflow(https://www.tensorflow.org/install/) 
  python(https://www.python.org/downloads/)
  
# 4. Usage of source code
  Download traning data and put it into the folder of "training data"
  Download VGG-16 model and put it into the folder of "VGG_MODEL"
  please be careful of the consistency of these names with the code. These may be some changes to make them consistency.

  ## 
  4.1 run test_CycleIR.py to test the images in the "test_image" folder. 
  
  ## 
  4.2 run train_CycleIR.py to training. The trained model is saved in the ckpt-wgan folder.
