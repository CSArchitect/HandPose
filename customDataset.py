import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import os
import os.path as osp
import numpy as np
import json
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import mltools as ml
from PIL import Image

import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

import glob
from pprint import pprint

# ================================================================== #
#                  Input pipeline for custom dataset                 #
# ================================================================== #

img_size = 368, 368
lbl_size = 21, 3
#path = 'test_batch_synth2/'
    
    
# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        # Initialize file paths 
#         self.paths = ['hand_labels_synth/synth2/',
#                       'hand_labels_synth/synth3/']
        self.paths = ['hand_labels_synth/test2/',
                      'hand_labels_synth/test3/']

        # Initialize first folder
        imgFiles_1 = []
        imgFiles_1 = glob.glob(osp.join(self.paths[0], '*.jpg'))
        numFiles_1 = len(imgFiles_1)
        
        jsonFiles_1 = []
        jsonFiles_1 = glob.glob(osp.join(self.paths[0], '*.json'))
        
        # Initialize second folder
        imgFiles_2 = []
        imgFiles_2 = glob.glob(osp.join(self.paths[1], '*.jpg'))
        numFiles_2 = len(imgFiles_2)
        
        jsonFiles_2 = []
        jsonFiles_2 = glob.glob(osp.join(self.paths[1], '*.json'))
        
        # Get total number of 
        numFiles = numFiles_1 + numFiles_2

        # Initialize a list of file names. 
        self.imgs = np.chararray(numFiles, itemsize=37)
        
        # Initialize images' labels
        self.labels = torch.zeros(numFiles,21,3)
        
        # Initialize images RGB pixels to calculate mean/std
        pixels = torch.zeros(numFiles,3,368,368)  

        pil2tensor = transforms.ToTensor()
        
        
        #################################################################
        
        for i,f in enumerate(imgFiles_1):
            self.imgs[i] = f

            pil_image = Image.open(f)
            rgb_image = pil2tensor(pil_image)*255

            pixels[i] = rgb_image
            pil_image.close()
            
        for i, f in enumerate(jsonFiles_1):
            with open(f, 'r') as fid:
                dat = json.load(fid)
            self.labels[i] = torch.Tensor(dat['hand_pts'])
            
        print("-----Synth 2 Loaded------")
        
        #################################################################

        for i,f in enumerate(imgFiles_2):
            self.imgs[i+numFiles_1] = f

            pil_image = Image.open(f)
            rgb_image = pil2tensor(pil_image)*255

            pixels[i] = rgb_image
            pil_image.close()
            
        for i, f in enumerate(jsonFiles_2):
            with open(f, 'r') as fid:
                dat = json.load(fid)
            self.labels[i+numFiles_1] = torch.Tensor(dat['hand_pts'])
    
        print("-----Synth 3 Loaded------")
        
        #################################################################

        # The total mean of all images per channel
        self.mean_channels = torch.mean(pixels, dim=0)
        self.std_channels = torch.std(pixels, dim=0)
#         print("Mean Shape: ", self.mean_channels.shape)
#         print("Mean: ", self.mean_channels)
#         print("Std Shape: ", self.std_channels.shape)
#         print("Std: ", self.std_channels)
        pprint(self.mean_channels[0])
        print("______________________________")
        pprint(self.mean_channels[1])
        print("______________________________")
        pprint(self.mean_channels[2])
        print("______________________________")
                
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pil2tensor = transforms.ToTensor()
        pil_image = Image.open(self.imgs[index])
        rgb_image = pil2tensor(pil_image)*255
        
        img_norm = TF.normalize(rgb_image, 
                                mean=self.mean_channels, std=self.std_channels)
        pil_img.close()
        # 3. Return a data pair (e.g. image and label).
        return (img_norm, self.labels[index])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)