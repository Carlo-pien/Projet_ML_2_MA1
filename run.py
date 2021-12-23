!nvidia-smi

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

!pip install einops
!pip install pytorch-lightning
!pip install -q -U segmentation-models-pytorch albumentations > /dev/null

%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import numpy as np
import math
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
import torchvision.transforms.functional as TF
from einops import rearrange, reduce
import pytorch_lightning as pl
import cv2
import segmentation_models_pytorch as smp
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")
import albumentations as album

#creating helper functions for image preprocessing and visualization
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def to_RGB(mask):
    return np.repeat(np.round(mask)[:, :, np.newaxis], 3, axis=2)


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.numpy().astype(int)]

    return x

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
    
#provided functions for loading the data    
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

#Dowloading the data
root_dir = "/content/drive/MyDrive/Projet_ML/training/"

n = 100
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(n, len(files))
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

data = []
for idx, img in enumerate(imgs) :
    data.append(img)

labels = []
for idx, mask in enumerate(gt_imgs) :
    labels.append(mask)

#Creation of our Dataset class
class RoadsDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            data = data,
            labels = labels,
            class_rgb_values=[[255, 255, 255], [0, 0, 0]], 
            augmentation=None, 
            preprocessing=None,
    ):

        self.image_ = data
        self.mask_ = labels
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = np.array(self.image_)[i,:,:,:]
        mask = np.array(self.mask_)[i,:,:]
        
        # one-hot-encode the mask
        mask = to_RGB(mask)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            
        return image, mask
    
    def __len__(self):
        # return length of 
        return np.array(data).shape[0]

def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.RandomSizedCrop(min_max_height=	[200,200], height = 400, width=400, p = 0.5),
        album.ShiftScaleRotate(p=0.5),
        album.Affine(shear=[-10, 10], scale=1.1, fit_output=False, p=.1),
        album.GaussNoise(var_limit=(.001, .03), p=.5),
        album.Blur(blur_limit=5., p=.2),
        album.ImageCompression(quality_lower=50, p=.2),
    ]
    return album.Compose(train_transform)

dataset = RoadsDataset(data, labels, augmentation=get_training_augmentation(), preprocessing=get_preprocessing())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


#loading the test set. Test images are resized to the 400 by 400 format of the training images
test_aug_downsize = album.RandomSizedCrop(min_max_height=	[608,608], height = 400, width=400, p = 1.)

root_dir_test = "/content/drive/MyDrive/Projet_ML/test_set_images/"
n_test = 50
files_test = os.listdir(root_dir_test)
files_test = [file for file in files_test if file.endswith('.png')]
n_test = min(n_test, len(files_test))
imgs_test = []
imgs_test_names = []
for i in range(n_test):
    imgs_test.append(test_aug_downsize(image=load_image(root_dir_test + files_test[i])))
    img_name = str(files_test[i])
    img_name = img_name.replace("test_", "")
    img_name = img_name.replace(".png", "")
    img_name = img_name.zfill(2)
    imgs_test_names.append(img_name)
    
#reloading the model
path = os.path.join('/content/drive/MyDrive/Projet_ML/pretrained_models/deeplabv3p_500_epochs.pth')
model = torch.load(path)
model.eval()


#generating predictions and storing the predicted masks in a list
test_aug_upsize = album.RandomSizedCrop(min_max_height=	[400,400], height = 608, width=608, p = 1.)

preds = []
for i in range(n_test):
    y = model(rearrange(torch.unsqueeze(torch.Tensor(imgs_test[i]['image']), dim=0), 'b h w c -> b c h w').to('cuda'))
    y = rearrange(torch.squeeze(y, dim=0).to('cpu').detach().numpy(), 'c h w -> h w c')
    y = test_aug_upsize(image=y)
    preds.append(y)

#creating a new list with only the relevant part of the masks (that is, the masks are dictionaries, so we evaluate them to get only the keys associated to the masks. 
for_submission = []
for i, pred in enumerate(preds) :
    for_submission.append(1-(preds[i]['image'].transpose(2, 0, 1)[1]))

#Saving the masks in a new directory
from PIL import Image as im
path = os.path.join('/content/drive/MyDrive/Projet_ML/for_submission/')

for i, mask in enumerate(for_submission):
    name = imgs_test_names[i]
    mask = im.fromarray(255*mask)
    mask = mask.convert('RGB')
    mask.save(path+name+'.png')
    
    
#Creation of a CSV submission file
import os
import numpy as np
import matplotlib.image as mpimg
import re

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df < foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = 1-patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def create_submission():
    submission_filename = '/content/drive/MyDrive/Projet_ML/submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = '/content/drive/MyDrive/Projet_ML/for_submission/' + '%.2d' % i + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

create_submission()