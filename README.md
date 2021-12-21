# EPFL Road Segmentation 2020

This repository contains the code used to create our submission to the 2020 edition of the [EPFL Aicrowd Road Segmentation challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

## Libraries used and Execution Guide


<details>
  <summary>Click to show / hide</summary>
<br>
  
All the notebooks included were run using google colab, we thus recommend google colab for their execution. Should a local alternative be desirable, however the following imports, along with a version of python of 3.6.9 (the one present on google colab) are required:

```
%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import math
from sklearn.model_selection import KFold
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as transforms
import albumentations.augmentations.transforms as tf
from torchvision.transforms import Compose
import albumentations as A
import torchvision.transforms.functional as TF
from einops import rearrange, reduce
import pytorch_lightning as pl
import random, tqdm
import seaborn as sns
import warnings
```


</details>

## Our usage of Google Colab

<details>
  <summary>Click to show / hide</summary>
<br>

In order to run all our experiments with good GPUs, we chose to use the Google Colab platform, thus, all our notebooks are hosted there. We also copied them to the github classroom for completeness (looking at code / outputs without running cells), but, since they all make use of google colab and google drive, to run them like we do, you need to follow these steps :

- Access this link that points to our Code Folder, named "Project_ML" : https://drive.google.com/drive/folders/1BdVy8AukS7MS5bqCxJuJMF2N-cUwKluq?usp=sharing
- Add a shortcut to the Code Folder inside your root drive (Right-click on the folder, add a shortcut inside Drive), without changing the name
- When running a notebook, make sure that the Drive mount folder shows our code folder inside /content/gdrive/MyDrive, otherwise, the shortcut either has the wrong name, or is at the wrong location
- Sometimes colab allocates you worse GPUs than necessary, so you may need to reconnect to another machine if you try to train a model and get an OutOfMemory error when allocating Tensors.

Here is a description of everything in our Code Folder :
- archive : We kept most previous versions of our notebooks for completeness in this folder
- libs : All python libraries are kept under this folder (even those provided)
- models : All pretrained weights and models are in this folder
- submissions : We kept csv files for all important submissions in this folder.
- test_predictions : We kept image predictions for all important submissions in this folder
- test_set_images : The folder of test images
- training : The folder of training images
- validation_predictions : The image predictions that are done on our validation sets in our Experiments notebook are kept here.
- vis_postprocessing : Figures appearing in the report relating to postprocessing.
- ipynb files : All notebooks are described later
- run.py file: Same as the run.py in the github classroom folder, here for ease of use of the Running.ipynb notebook

</details>

## Documentation of our solution

<details>
  <summary>Click to show / hide</summary>
<br>
  
### The run.ipynb file

The `run.ipynb` file performs the following steps :

- Downloading our best model in the drive (i.e. 500 training epochs using all transformations for the data augmentation)
- Predicting the test images
- Creating a submission file for the AIcrowd platform

It must be run with GPUs and might not work and might not work with GPUs that have a lower amount of memory than the ones on *Google Colab*.
It is also possible to train a model using the same parameters as our best model thanks to the `pretrained.ipynb` file, instead of downloading the best model, but it takes lots of hours to run it on ***Google Colab***.
  
### The nn.ipynb file
  
The `nn.ipynb` file is the one that allow us to train our Unet model by changing different parameters such as the transformations and the number of epochs and then creating a submission thanks to it.

You can change the different transformations parameters by commenting them or not in the class *CustomDataset*. Moreover, the number of epochs can be changed in the paragraph where we define the model.
  
This must be run on Google Colab on our drive to have access to the dataset, to the images of the test set and also to the different trained model to the                     submission.
  
### The pretrained.ipynb file
  
The `pretrained.ipynb` file is the one that allow us to train the *DeepLabV3Plus* model by changing different parameters such as the transformations and the number of epochs and then creating a submission thanks to it.
You can change the different transformations parameters by commenting them or not in the class *RoadsDataset*. Moreover, the number of epochs can be changed in the paragraph where we define the model.
  
This must be run on Google Colab on our drive to have access to the dataset, to the images of the test set and also to the different trained model to the                     submission.

### The Experiments Notebook

The `Experiments.ipynb` notebook combines most of our experiments attempted on this project. 

Most cells have to be run on Google Colab or at least using similar/better GPUs (Nvidia K80 at least, but we can't guarantee that they didn't change since), although we don't even necessarily recommend running them, because training models can take multiple hours. Everything is already run, with shown output so that you can look at code and corresponding output.

It is divided into parts which are :

- Selecting Data Augmentation
- Selecting the level of our U-Net
- Best Input Size (and sliding window size)
- Best stride for our sliding window
- Trying out weighted loss
- Averaging models
- Post-Processing methods to use


### The file libraries

In order to tidy up code inside the notebooks, we chose to move all shared / boilerplate code inside different python files which we use as libraries (listed under the libs folder).

  
#### Unet_model.py

This file contains a first version of our Unet neural network without using the *Lightning* library. Thus, this ones is too slow to run and we didn't use it for our experimentations.
  
#### Unet_model_lightning.py

This file contains the second version of our Unet neural network using the *Lightning* library. 

This model is the one we used for all our predictions made with a Unet.

</details>

----

### Authors :

- Ghenassia Noam
- Nussbaumer Arthur
- Piening Carlo
