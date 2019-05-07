# Cycle-IR: Deep Cyclic Image Retargeting
# 1. Overview

This is a Tensorflow implementation of Cycle-IR approach for content-aware image retargeting.

![](https://github.com/mintanwei/Cycle-IR/blob/master/AdjustmentOfAspectRatio.png)

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

  If you have any question to use this code, please be feel free to contact me (wmtan14@fudan.edu.cn).
