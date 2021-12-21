{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi\n",
    "\n",
    "from google.colab import drive\n",
    "drive.mount(\"/content/drive\", force_remount=True)\n",
    "\n",
    "!pip install einops\n",
    "!pip install pytorch-lightning\n",
    "!pip install -q -U segmentation-models-pytorch albumentations > /dev/null\n",
    "\n",
    "%matplotlib inline\n",
    "import matplotlib.image as mpimg\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import os,sys\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import math\n",
    "import torch\n",
    "import torch.autograd as autograd\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.autograd import Variable\n",
    "from torch.utils.data import Dataset\n",
    "from torch.utils.data import TensorDataset, DataLoader\n",
    "from torchvision.io import read_image\n",
    "from torchvision import transforms as transforms\n",
    "import albumentations.augmentations.transforms as tf\n",
    "from torchvision.transforms import Compose\n",
    "import torchvision.transforms.functional as TF\n",
    "from einops import rearrange, reduce\n",
    "import pytorch_lightning as pl\n",
    "import cv2\n",
    "import segmentation_models_pytorch as smp\n",
    "import os, cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import random, tqdm\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import albumentations as album\n",
    "\n",
    "#creating helper functions for image preprocessing and visualization\n",
    "def one_hot_encode(label, label_values):\n",
    "    \"\"\"\n",
    "    Convert a segmentation image label array to one-hot format\n",
    "    by replacing each pixel value with a vector of length num_classes\n",
    "    # Arguments\n",
    "        label: The 2D array segmentation image label\n",
    "        label_values\n",
    "        \n",
    "    # Returns\n",
    "        A 2D array with the same width and hieght as the input, but\n",
    "        with a depth size of num_classes\n",
    "    \"\"\"\n",
    "    semantic_map = []\n",
    "    for colour in label_values:\n",
    "        equality = np.equal(label, colour)\n",
    "        class_map = np.all(equality, axis = -1)\n",
    "        semantic_map.append(class_map)\n",
    "    semantic_map = np.stack(semantic_map, axis=-1)\n",
    "\n",
    "    return semantic_map\n",
    "\n",
    "\n",
    "def to_RGB(mask):\n",
    "    return np.repeat(np.round(mask)[:, :, np.newaxis], 3, axis=2)\n",
    "\n",
    "\n",
    "def reverse_one_hot(image):\n",
    "    \"\"\"\n",
    "    Transform a 2D array in one-hot format (depth is num_classes),\n",
    "    to a 2D array with only 1 channel, where each pixel value is\n",
    "    the classified class key.\n",
    "    # Arguments\n",
    "        image: The one-hot format image \n",
    "        \n",
    "    # Returns\n",
    "        A 2D array with the same width and hieght as the input, but\n",
    "        with a depth size of 1, where each pixel value is the classified \n",
    "        class key.\n",
    "    \"\"\"\n",
    "    x = np.argmax(image, axis = -1)\n",
    "    return x\n",
    "\n",
    "def colour_code_segmentation(image, label_values):\n",
    "    \"\"\"\n",
    "    Given a 1-channel array of class keys, colour code the segmentation results.\n",
    "    # Arguments\n",
    "        image: single channel array where each value represents the class key.\n",
    "        label_values\n",
    "\n",
    "    # Returns\n",
    "        Colour coded image for segmentation visualization\n",
    "    \"\"\"\n",
    "    colour_codes = np.array(label_values)\n",
    "    x = colour_codes[image.numpy().astype(int)]\n",
    "\n",
    "    return x\n",
    "\n",
    "def to_tensor(x, **kwargs):\n",
    "    return x.transpose(2, 0, 1).astype('float32')\n",
    "\n",
    "def get_preprocessing(preprocessing_fn=None):\n",
    "    \"\"\"Construct preprocessing transform    \n",
    "    Args:\n",
    "        preprocessing_fn (callable): data normalization function \n",
    "            (can be specific for each pretrained neural network)\n",
    "    Return:\n",
    "        transform: albumentations.Compose\n",
    "    \"\"\"\n",
    "    _transform = []\n",
    "    if preprocessing_fn:\n",
    "        _transform.append(album.Lambda(image=preprocessing_fn))\n",
    "    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))\n",
    "        \n",
    "    return album.Compose(_transform)\n",
    "\n",
    "def visualize(**images):\n",
    "    \"\"\"\n",
    "    Plot images in one row\n",
    "    \"\"\"\n",
    "    n_images = len(images)\n",
    "    plt.figure(figsize=(20,8))\n",
    "    for idx, (name, image) in enumerate(images.items()):\n",
    "        plt.subplot(1, n_images, idx + 1)\n",
    "        plt.xticks([]); \n",
    "        plt.yticks([])\n",
    "        # get title from the parameter names\n",
    "        plt.title(name.replace('_',' ').title(), fontsize=20)\n",
    "        plt.imshow(image)\n",
    "    plt.show()\n",
    "    \n",
    "#provided functions for loading the data    \n",
    "def load_image(infilename):\n",
    "    data = mpimg.imread(infilename)\n",
    "    return data\n",
    "\n",
    "def img_float_to_uint8(img):\n",
    "    rimg = img - np.min(img)\n",
    "    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)\n",
    "    return rimg\n",
    "\n",
    "# Concatenate an image and its groundtruth\n",
    "def concatenate_images(img, gt_img):\n",
    "    nChannels = len(gt_img.shape)\n",
    "    w = gt_img.shape[0]\n",
    "    h = gt_img.shape[1]\n",
    "    if nChannels == 3:\n",
    "        cimg = np.concatenate((img, gt_img), axis=1)\n",
    "    else:\n",
    "        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)\n",
    "        gt_img8 = img_float_to_uint8(gt_img)          \n",
    "        gt_img_3c[:,:,0] = gt_img8\n",
    "        gt_img_3c[:,:,1] = gt_img8\n",
    "        gt_img_3c[:,:,2] = gt_img8\n",
    "        img8 = img_float_to_uint8(img)\n",
    "        cimg = np.concatenate((img8, gt_img_3c), axis=1)\n",
    "    return cimg\n",
    "\n",
    "def img_crop(im, w, h):\n",
    "    list_patches = []\n",
    "    imgwidth = im.shape[0]\n",
    "    imgheight = im.shape[1]\n",
    "    is_2d = len(im.shape) < 3\n",
    "    for i in range(0,imgheight,h):\n",
    "        for j in range(0,imgwidth,w):\n",
    "            if is_2d:\n",
    "                im_patch = im[j:j+w, i:i+h]\n",
    "            else:\n",
    "                im_patch = im[j:j+w, i:i+h, :]\n",
    "            list_patches.append(im_patch)\n",
    "    return list_patches\n",
    "\n",
    "#Dowloading the data\n",
    "root_dir = \"/content/drive/MyDrive/Projet_ML/training/\"\n",
    "\n",
    "n = 100\n",
    "image_dir = root_dir + \"images/\"\n",
    "files = os.listdir(image_dir)\n",
    "n = min(n, len(files))\n",
    "print(\"Loading \" + str(n) + \" images\")\n",
    "imgs = [load_image(image_dir + files[i]) for i in range(n)]\n",
    "\n",
    "gt_dir = root_dir + \"groundtruth/\"\n",
    "print(\"Loading \" + str(n) + \" images\")\n",
    "gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]\n",
    "\n",
    "data = []\n",
    "for idx, img in enumerate(imgs) :\n",
    "    data.append(img)\n",
    "\n",
    "labels = []\n",
    "for idx, mask in enumerate(gt_imgs) :\n",
    "    labels.append(mask)\n",
    "\n",
    "#Creation of our Dataset class\n",
    "class RoadsDataset(torch.utils.data.Dataset):\n",
    "\n",
    "    def __init__(\n",
    "            self, \n",
    "            data = data,\n",
    "            labels = labels,\n",
    "            class_rgb_values=[[255, 255, 255], [0, 0, 0]], \n",
    "            augmentation=None, \n",
    "            preprocessing=None,\n",
    "    ):\n",
    "\n",
    "        self.image_ = data\n",
    "        self.mask_ = labels\n",
    "        \n",
    "        self.class_rgb_values = class_rgb_values\n",
    "        self.augmentation = augmentation\n",
    "        self.preprocessing = preprocessing\n",
    "    \n",
    "    def __getitem__(self, i):\n",
    "        \n",
    "        # read images and masks\n",
    "        image = np.array(self.image_)[i,:,:,:]\n",
    "        mask = np.array(self.mask_)[i,:,:]\n",
    "        \n",
    "        # one-hot-encode the mask\n",
    "        mask = to_RGB(mask)\n",
    "        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')\n",
    "        \n",
    "        # apply augmentations\n",
    "        if self.augmentation:\n",
    "            sample = self.augmentation(image=image, mask=mask)\n",
    "            image, mask = sample['image'], sample['mask']\n",
    "        \n",
    "        # apply preprocessing\n",
    "        if self.preprocessing:\n",
    "            sample = self.preprocessing(image=image, mask=mask)\n",
    "            image, mask = sample['image'], sample['mask']\n",
    "\n",
    "            \n",
    "        return image, mask\n",
    "    \n",
    "    def __len__(self):\n",
    "        # return length of \n",
    "        return np.array(data).shape[0]\n",
    "\n",
    "def get_training_augmentation():\n",
    "    train_transform = [\n",
    "        album.HorizontalFlip(p=0.5),\n",
    "        album.VerticalFlip(p=0.5),\n",
    "        album.RandomSizedCrop(min_max_height=\t[200,200], height = 400, width=400, p = 0.5),\n",
    "        album.ShiftScaleRotate(p=0.5),\n",
    "        album.Affine(shear=[-10, 10], scale=1.1, fit_output=False, p=.1),\n",
    "        album.GaussNoise(var_limit=(.001, .03), p=.5),\n",
    "        album.Blur(blur_limit=5., p=.2),\n",
    "        album.ImageCompression(quality_lower=50, p=.2),\n",
    "    ]\n",
    "    return album.Compose(train_transform)\n",
    "\n",
    "dataset = RoadsDataset(data, labels, augmentation=get_training_augmentation(), preprocessing=get_preprocessing())\n",
    "dataloader = DataLoader(dataset, batch_size=16, shuffle=True)\n",
    "\n",
    "\n",
    "#loading the test set. Test images are resized to the 400 by 400 format of the training images\n",
    "test_aug_downsize = album.RandomSizedCrop(min_max_height=\t[608,608], height = 400, width=400, p = 1.)\n",
    "\n",
    "root_dir_test = \"/content/drive/MyDrive/Projet_ML/test_set_images/\"\n",
    "n_test = 50\n",
    "files_test = os.listdir(root_dir_test)\n",
    "files_test = [file for file in files_test if file.endswith('.png')]\n",
    "n_test = min(n_test, len(files_test))\n",
    "imgs_test = []\n",
    "imgs_test_names = []\n",
    "for i in range(n_test):\n",
    "    imgs_test.append(test_aug_downsize(image=load_image(root_dir_test + files_test[i])))\n",
    "    img_name = str(files_test[i])\n",
    "    img_name = img_name.replace(\"test_\", \"\")\n",
    "    img_name = img_name.replace(\".png\", \"\")\n",
    "    img_name = img_name.zfill(2)\n",
    "    imgs_test_names.append(img_name)\n",
    "    \n",
    "#reloading the model\n",
    "path = os.path.join('/content/drive/MyDrive/Projet_ML/pretrained_models/deeplabv3p_500_epochs.pth')\n",
    "model = torch.load(path)\n",
    "model.eval()\n",
    "\n",
    "\n",
    "#generating predictions and storing the predicted masks in a list\n",
    "test_aug_upsize = album.RandomSizedCrop(min_max_height=\t[400,400], height = 608, width=608, p = 1.)\n",
    "\n",
    "preds = []\n",
    "for i in range(n_test):\n",
    "    y = model(rearrange(torch.unsqueeze(torch.Tensor(imgs_test[i]['image']), dim=0), 'b h w c -> b c h w').to('cuda'))\n",
    "    y = rearrange(torch.squeeze(y, dim=0).to('cpu').detach().numpy(), 'c h w -> h w c')\n",
    "    y = test_aug_upsize(image=y)\n",
    "    preds.append(y)\n",
    "\n",
    "#creating a new list with only the relevant part of the masks (that is, the masks are dictionaries, so we evaluate them to get only the keys associated to the masks. \n",
    "for_submission = []\n",
    "for i, pred in enumerate(preds) :\n",
    "    for_submission.append(1-(preds[i]['image'].transpose(2, 0, 1)[1]))\n",
    "\n",
    "#Saving the masks in a new directory\n",
    "from PIL import Image as im\n",
    "path = os.path.join('/content/drive/MyDrive/Projet_ML/for_submission/')\n",
    "\n",
    "for i, mask in enumerate(for_submission):\n",
    "    name = imgs_test_names[i]\n",
    "    mask = im.fromarray(255*mask)\n",
    "    mask = mask.convert('RGB')\n",
    "    mask.save(path+name+'.png')\n",
    "    \n",
    "    \n",
    "#Creation of a CSV submission file\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.image as mpimg\n",
    "import re\n",
    "\n",
    "foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch\n",
    "\n",
    "# assign a label to a patch\n",
    "def patch_to_label(patch):\n",
    "    df = np.mean(patch)\n",
    "    if df < foreground_threshold:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "\n",
    "\n",
    "def mask_to_submission_strings(image_filename):\n",
    "    \"\"\"Reads a single image and outputs the strings that should go into the submission file\"\"\"\n",
    "    img_number = int(re.search(r\"\\d+\", image_filename).group(0))\n",
    "    im = mpimg.imread(image_filename)\n",
    "    patch_size = 16\n",
    "    for j in range(0, im.shape[1], patch_size):\n",
    "        for i in range(0, im.shape[0], patch_size):\n",
    "            patch = im[i:i + patch_size, j:j + patch_size]\n",
    "            label = 1-patch_to_label(patch)\n",
    "            yield(\"{:03d}_{}_{},{}\".format(img_number, j, i, label))\n",
    "\n",
    "\n",
    "def masks_to_submission(submission_filename, *image_filenames):\n",
    "    \"\"\"Converts images into a submission file\"\"\"\n",
    "    with open(submission_filename, 'w') as f:\n",
    "        f.write('id,prediction\\n')\n",
    "        for fn in image_filenames[0:]:\n",
    "            f.writelines('{}\\n'.format(s) for s in mask_to_submission_strings(fn))\n",
    "\n",
    "\n",
    "def create_submission():\n",
    "    submission_filename = '/content/drive/MyDrive/Projet_ML/submission.csv'\n",
    "    image_filenames = []\n",
    "    for i in range(1, 51):\n",
    "        image_filename = '/content/drive/MyDrive/Projet_ML/for_submission/' + '%.2d' % i + '.png'\n",
    "        image_filenames.append(image_filename)\n",
    "    masks_to_submission(submission_filename, *image_filenames)\n",
    "\n",
    "create_submission()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
