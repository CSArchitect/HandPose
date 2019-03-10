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

# ================================================================== #
#                  Input pipeline for custom dataset                 #
# ================================================================== #

img_size = 368, 368
lbl_size = 21, 3
#path = 'test_batch_synth2/'
    
    
# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        # TODO
        # Initialize file paths 
        #self.paths = ['hand_labels_synth/synth2/', 'hand_labels_synth/synth3/']
        self.paths = ['hand_labels_synth\synth2/', 'test_batch_synth3/']
        # Initialize a list of file names. 
        self.imgs = np.chararray(3243, itemsize=37)
        
        # Initialize images RGB pixels to calculate mean/std
        # 5591 images (TODO: Count how many jpg's there are to avoid
        #                    hard-coded values)
        # 368*368 = 135424 pixels
        # 3 channels for RGB
        pixels = torch.zeros(3243,3,368,368)

        # Initialize images' RGB means
#         sum_pixels = torch.zeros(3,368,368)
        
        # Initialize images' RGB std
#         stds = torch.zeros(3,368,368)
        
        # Initialize images' labels
        self.labels = torch.zeros(3243,21,3)
                           
            
        inpath2 = self.paths[0]   
        # Used to pick up where synth2 left off before all
        # 5591 images
        cutoff = 0
        imgFiles = []
        imgFiles = glob.glob(osp.join(self.paths[0], '*.jpg'))
        for i,f in enumerate(imgFiles):
            # Since every iteration can be a .jpg or a .json
            # index to add to is floor(i/2)
            index = i #int(np.floor(i/2))

            self.imgs[index] = f
            cur_img = Image.open(self.imgs[index])
            # Transposed to be able to index each channel easier
            cur_img_px = torch.transpose(torch.Tensor(cur_img.getdata()), 0, 1)
            cur_img_px = cur_img_px.view(3,368,368)
            pixels[index] = cur_img_px
#                 sum_pixels += pixels 
#                 stds += pixels
            cur_img.close()

#                 print("Image: ", self.imgs[index].decode('utf-8', "ignore"))
#                 print("Pixels: ", list(self.pixels[index].shape))
#                 print("Mean: ", means[index])
#                 print("Std: ", stds[index])

        jsonFiles = []
        jsonFiles = glob.glob(osp.join(self.paths[0], '*.json'))
        for i, f in enumerate(jsonFiles):
            with open(f, 'r') as fid:
                dat = json.load(fid)
            index = i
            self.labels[index] = torch.Tensor(dat['hand_pts'])
# #                 print("Labels: ", list(self.labels[index].shape))
# #                 print("_____________________________")
            cutoff = i
    
    
        print("-----Synth 2 Loaded------")
        
        """
        inpath3 = self.paths[1]
        cutoff = int((cutoff+1)/2)
        for i,f in enumerate(sorted(os.listdir(inpath3))):
            index = int(np.floor(i/2))
            
            if f.endswith('.jpg'):
                self.imgs[index] = inpath3+f
                cur_img = Image.open(self.imgs[index])
                cur_img_px = torch.transpose(torch.Tensor(cur_img.getdata()), 0, 1)
                cur_img_px = cur_img_px.view(3,368,368)
                pixels[index] = cur_img_px
#                 sum_pixels += pixels 
#                 stds += pixels
                cur_img.close()
                
#                 print("Image: ", self.imgs[index].decode('utf-8', "ignore"))
#                 print("Pixels: ", list(self.pixels[index].shape))
#                 print("Mean: ", means[index])
#                 print("Std: ", stds[index])

            else:
                with open(inpath3+f, 'r') as fid:
                    dat = json.load(fid)
                self.labels[index] = torch.Tensor(dat['hand_pts'])
# #                 print("Labels: ", list(self.labels[index].shape))
# #                 print("_____________________________")


        print("-----Synth 3 Loaded------")
    """
        # The total mean of all images per channel
        self.mean_channels = torch.mean(pixels, dim=0)
        self.std_channels = torch.std(pixels, dim=0)
        print("Mean Shape: ", self.mean_channels.shape)
        print("Mean: ", self.mean_channels)
        print("Std Shape: ", self.std_channels.shape)
        print("Std: ", self.std_channels)
                
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pil2tensor = transforms.ToTensor()
        pil_image = Image.open(self.imgs[index])
        rgb_image = pil2tensor(pil_image)
        img_norm = TF.normalize(rgb_image, mean=self.mean_channels, std=self.std_channels)
#         img.close()
        # 3. Return a data pair (e.g. image and label).
        return (img_norm, self.labels[index])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)