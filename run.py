{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Carlo-pien/Projet_ML_2_MA1/blob/master/run.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fKgM8dKX1AWB",
        "outputId": "ca845b65-a604-4eb8-f2a9-98cdaec7702f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mon Dec 20 16:02:46 2021       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 495.44       Driver Version: 460.32.03    CUDA Version: 11.2     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   40C    P0    61W / 149W |   1456MiB / 11441MiB |      0%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "|  No running processes found                                                 |\n",
            "+-----------------------------------------------------------------------------+\n"
          ]
        }
      ],
      "source": [
        "!nvidia-smi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Cnl_zEQVkgXq",
        "outputId": "4136d93e-3c9e-4e29-82e8-84e9f6101c3e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount(\"/content/drive\", force_remount=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dC56abQA4jr0",
        "outputId": "cf08abf8-1b89-43c5-b4c4-81b364be3cae"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting einops\n",
            "  Downloading einops-0.3.2-py3-none-any.whl (25 kB)\n",
            "Installing collected packages: einops\n",
            "Successfully installed einops-0.3.2\n",
            "Collecting pytorch-lightning\n",
            "  Downloading pytorch_lightning-1.5.6-py3-none-any.whl (525 kB)\n",
            "\u001b[K     |████████████████████████████████| 525 kB 5.2 MB/s \n",
            "\u001b[?25hCollecting pyDeprecate==0.3.1\n",
            "  Downloading pyDeprecate-0.3.1-py3-none-any.whl (10 kB)\n",
            "Collecting PyYAML>=5.1\n",
            "  Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)\n",
            "\u001b[K     |████████████████████████████████| 596 kB 32.7 MB/s \n",
            "\u001b[?25hRequirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (4.62.3)\n",
            "Collecting torchmetrics>=0.4.1\n",
            "  Downloading torchmetrics-0.6.2-py3-none-any.whl (332 kB)\n",
            "\u001b[K     |████████████████████████████████| 332 kB 43.9 MB/s \n",
            "\u001b[?25hCollecting future>=0.17.1\n",
            "  Downloading future-0.18.2.tar.gz (829 kB)\n",
            "\u001b[K     |████████████████████████████████| 829 kB 47.0 MB/s \n",
            "\u001b[?25hCollecting fsspec[http]!=2021.06.0,>=2021.05.0\n",
            "  Downloading fsspec-2021.11.1-py3-none-any.whl (132 kB)\n",
            "\u001b[K     |████████████████████████████████| 132 kB 43.5 MB/s \n",
            "\u001b[?25hRequirement already satisfied: tensorboard>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (2.7.0)\n",
            "Requirement already satisfied: numpy>=1.17.2 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (1.19.5)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (3.10.0.2)\n",
            "Requirement already satisfied: packaging>=17.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (21.3)\n",
            "Requirement already satisfied: torch>=1.7.* in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning) (1.10.0+cu111)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (2.23.0)\n",
            "Collecting aiohttp\n",
            "  Downloading aiohttp-3.8.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)\n",
            "\u001b[K     |████████████████████████████████| 1.1 MB 43.8 MB/s \n",
            "\u001b[?25hRequirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=17.0->pytorch-lightning) (3.0.6)\n",
            "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (0.4.6)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (1.35.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (3.3.6)\n",
            "Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (3.17.3)\n",
            "Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (0.37.0)\n",
            "Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (0.6.1)\n",
            "Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (57.4.0)\n",
            "Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (1.0.1)\n",
            "Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (1.42.0)\n",
            "Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (0.12.0)\n",
            "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning) (1.8.0)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from absl-py>=0.4->tensorboard>=2.2.0->pytorch-lightning) (1.15.0)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning) (0.2.8)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning) (4.8)\n",
            "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning) (4.2.4)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning) (1.3.0)\n",
            "Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard>=2.2.0->pytorch-lightning) (4.8.2)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard>=2.2.0->pytorch-lightning) (3.6.0)\n",
            "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning) (0.4.8)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (2.10)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (2021.10.8)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (1.24.3)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning) (3.1.1)\n",
            "Collecting async-timeout<5.0,>=4.0.0a3\n",
            "  Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (21.2.0)\n",
            "Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->fsspec[http]!=2021.06.0,>=2021.05.0->pytorch-lightning) (2.0.8)\n",
            "Collecting aiosignal>=1.1.2\n",
            "  Downloading aiosignal-1.2.0-py3-none-any.whl (8.2 kB)\n",
            "Collecting multidict<7.0,>=4.5\n",
            "  Downloading multidict-5.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (160 kB)\n",
            "\u001b[K     |████████████████████████████████| 160 kB 45.8 MB/s \n",
            "\u001b[?25hCollecting yarl<2.0,>=1.0\n",
            "  Downloading yarl-1.7.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (271 kB)\n",
            "\u001b[K     |████████████████████████████████| 271 kB 34.6 MB/s \n",
            "\u001b[?25hCollecting frozenlist>=1.1.1\n",
            "  Downloading frozenlist-1.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (192 kB)\n",
            "\u001b[K     |████████████████████████████████| 192 kB 40.1 MB/s \n",
            "\u001b[?25hCollecting asynctest==0.13.0\n",
            "  Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)\n",
            "Building wheels for collected packages: future\n",
            "  Building wheel for future (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for future: filename=future-0.18.2-py3-none-any.whl size=491070 sha256=fe584c10a6522d66c1ea7082bf8f07edd98b32ea0dd16c470c552c738440b66d\n",
            "  Stored in directory: /root/.cache/pip/wheels/56/b0/fe/4410d17b32f1f0c3cf54cdfb2bc04d7b4b8f4ae377e2229ba0\n",
            "Successfully built future\n",
            "Installing collected packages: multidict, frozenlist, yarl, asynctest, async-timeout, aiosignal, fsspec, aiohttp, torchmetrics, PyYAML, pyDeprecate, future, pytorch-lightning\n",
            "  Attempting uninstall: PyYAML\n",
            "    Found existing installation: PyYAML 3.13\n",
            "    Uninstalling PyYAML-3.13:\n",
            "      Successfully uninstalled PyYAML-3.13\n",
            "  Attempting uninstall: future\n",
            "    Found existing installation: future 0.16.0\n",
            "    Uninstalling future-0.16.0:\n",
            "      Successfully uninstalled future-0.16.0\n",
            "Successfully installed PyYAML-6.0 aiohttp-3.8.1 aiosignal-1.2.0 async-timeout-4.0.2 asynctest-0.13.0 frozenlist-1.2.0 fsspec-2021.11.1 future-0.18.2 multidict-5.2.0 pyDeprecate-0.3.1 pytorch-lightning-1.5.6 torchmetrics-0.6.2 yarl-1.7.2\n"
          ]
        }
      ],
      "source": [
        "!pip install einops\n",
        "!pip install pytorch-lightning\n",
        "!pip install -q -U segmentation-models-pytorch albumentations > /dev/null"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wctuy3BGJqjK"
      },
      "source": [
        "Importing the relevant packages"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Yw-WjEW_4fYQ"
      },
      "outputs": [],
      "source": [
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
        "import albumentations as album"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "F2MQ4BJSJvUC"
      },
      "source": [
        "creating helper functions for image preprocessing and visualization"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7FIzH6cbmzEf"
      },
      "outputs": [],
      "source": [
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
        "  return np.repeat(np.round(mask)[:, :, np.newaxis], 3, axis=2)\n",
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
        "    plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_b0GLD4ZJ45q"
      },
      "source": [
        "provided functions for loading the data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mB29ysam4Icg"
      },
      "outputs": [],
      "source": [
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
        "    return list_patches"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CS4Iv8QdlcYY"
      },
      "outputs": [],
      "source": [
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
        "    return album.Compose(train_transform)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "J8otWt7gKXTT"
      },
      "source": [
        "loading the data in the dataset, and creating a dataloader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0y5DcZKv6Tt6"
      },
      "outputs": [],
      "source": [
        "dataset = RoadsDataset(data, labels, augmentation=get_training_augmentation(), preprocessing=get_preprocessing())\n",
        "dataloader = DataLoader(dataset, batch_size=16, shuffle=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wkZiWICPL4dC"
      },
      "source": [
        "## predictions"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FCPCZ06sL8hr"
      },
      "source": [
        "loading the test set. Test images are resized to the 400 by 400 format of the training images"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2cSucTiOYxiu"
      },
      "outputs": [],
      "source": [
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
        "  imgs_test.append(test_aug_downsize(image=load_image(root_dir_test + files_test[i])))\n",
        "  img_name = str(files_test[i])\n",
        "  img_name = img_name.replace(\"test_\", \"\")\n",
        "  img_name = img_name.replace(\".png\", \"\")\n",
        "  img_name = img_name.zfill(2)\n",
        "  imgs_test_names.append(img_name)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aB491XIUVcZa"
      },
      "source": [
        "reloading the model (this is usefull if the runtime was stopped/restared after training, which might be necessary to save cuda memory)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iRQVJh8MDOpU"
      },
      "outputs": [],
      "source": [
        "path = os.path.join('/content/drive/MyDrive/Projet_ML/pretrained_models/deeplabv3p_500_epochs.pth')\n",
        "model = torch.load(path)\n",
        "model.eval()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FrCFyilqMiVJ"
      },
      "source": [
        "generating predictions and storing the predicted masks in a list"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1-L4Lrwc6i2N"
      },
      "outputs": [],
      "source": [
        "test_aug_upsize = album.RandomSizedCrop(min_max_height=\t[400,400], height = 608, width=608, p = 1.)\n",
        "\n",
        "preds = []\n",
        "for i in range(n_test):\n",
        "  y = model(rearrange(torch.unsqueeze(torch.Tensor(imgs_test[i]['image']), dim=0), 'b h w c -> b c h w').to('cuda'))\n",
        "  y = rearrange(torch.squeeze(y, dim=0).to('cpu').detach().numpy(), 'c h w -> h w c')\n",
        "  y = test_aug_upsize(image=y)\n",
        "  preds.append(y)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1XW0K_o4M7Ov"
      },
      "source": [
        "creating a new list with only the relevant part of the masks (that is, the masks are dictionaries, so we evaluate them to get only the keys associated to the masks. "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dSqut6HTaEpP"
      },
      "outputs": [],
      "source": [
        "for_submission = []\n",
        "for i, pred in enumerate(preds) :\n",
        "  #for_submission.append(1-(preds[i]['image'].transpose(2, 0, 1)[1])>0.8)\n",
        "  for_submission.append(1-(preds[i]['image'].transpose(2, 0, 1)[1]))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uDDGeG33Nh-A"
      },
      "source": [
        "saving the masks in a new directory"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nxN6qHmIk5zb"
      },
      "outputs": [],
      "source": [
        "from PIL import Image as im\n",
        "path = os.path.join('/content/drive/MyDrive/Projet_ML/for_submission/')\n",
        "\n",
        "for i, mask in enumerate(for_submission):\n",
        "  name = imgs_test_names[i]\n",
        "  mask = im.fromarray(255*mask)\n",
        "  mask = mask.convert('RGB')\n",
        "  mask.save(path+name+'.png')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Hadf0Y8wNnvr"
      },
      "source": [
        "creating a submission csv file"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IhuQz_lT0Wt2"
      },
      "outputs": [],
      "source": [
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
        "    #if df > foreground_threshold:\n",
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
        "            #label = patch_to_label(patch)\n",
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
        "  submission_filename = '/content/drive/MyDrive/Projet_ML/submission.csv'\n",
        "  image_filenames = []\n",
        "  for i in range(1, 51):\n",
        "      image_filename = '/content/drive/MyDrive/Projet_ML/for_submission/' + '%.2d' % i + '.png'\n",
        "      #print(image_filename)\n",
        "      image_filenames.append(image_filename)\n",
        "  masks_to_submission(submission_filename, *image_filenames)\n",
        "\n",
        "create_submission()\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "run.py",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}