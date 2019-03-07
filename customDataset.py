import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
import os.path
import numpy as np
import json
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import mltools as ml
from PIL import Image


# ================================================================== #
#                  Input pipeline for custom dataset                 #
# ================================================================== #

img_size = 368, 368
lbl_size = 21, 3

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
#         self.edges = [[0,1],[1,2],[2,3],[3,4],[0,5],
#                       [5,6],[6,7],[7,8],[0,9],[9,10],
#                       [10,11],[11,12],[0,13],[13,14],[14,15],
#                       [15,16],[0,17],[17,18],[18,19],[19,20]]

#         self.paths = ['synth1/', 'synth2/', 'synth3/', 'synth4/']
        self.paths = ['hand_labels_synth/synth2/', 'hand_labels_synth/synth3/']

        # files contain the 00000001.json - 00003243.json files sorted
#         files2 = sorted([self.paths[0]+f for f in os.listdir(self.paths[0]) if f.endswith('.jpg')])
#         files3 = sorted([self.paths[1]+f for f in os.listdir(self.paths[1]) if f.endswith('.jpg')])
        
#         self.imgs = np.chararray(5591)
        self.imgs = []
        self.labels = np.zeros((5591,21,3))
                               
        inpath2 = self.paths[0]
        cutoff = 0
        for i,f in enumerate(sorted(os.listdir(inpath2))):
            index = int(np.floor(i/2))
            
            if f.endswith('.jpg'):
                self.imgs.append((inpath2+f).encode('utf-8'))
#                 print(self.imgs[index].decode('utf-8', "ignore"))
            else:
                with open(inpath2+f, 'r') as fid:
                    dat = json.load(fid)
                self.labels[index] = np.array(dat['hand_pts'])
#                 print(self.labels[index])
#                 print("_____________________________")
            cutoff = i
    
        inpath3 = self.paths[1]
        cutoff = int((cutoff+1)/2)
        for i,f in enumerate(sorted(os.listdir(inpath3))):
            index = cutoff + int(np.floor(i/2))
            
            if f.endswith('.jpg'):
                self.imgs.append((inpath3+f).encode('utf-8'))
#                 print(self.imgs[index].decode('utf-8', "ignore"))
            else:
                with open(inpath3+f, 'r') as fid:
                    dat = json.load(fid)
                self.labels[index] = np.array(dat['hand_pts'])
#                 print(self.labels[index])

        print(len(self.imgs))
        print(np.shape(self.labels))
#         print("________________________________")
                
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        #img = Image.open(self.imgs[index])
        #pixels = list(img.getdata())
        #m = np.mean(pixels)
        #s = np.std(pixels)
        #normalize = transforms.Normalize(mean=m,std=s)
        
        #transform = transforms.Compose([transforms.ToTensor(), normalize])
        #img1 = transform(img)
        # plt.imshow(img)]
        
        pil2tensor = transforms.ToTensor()
        #tensor2pil = transforms.ToPILImage()
        
        pil_image = Image.open(self.imgs[index])
        rgb_image = pil2tensor(pil_image)
        # 3. Return a data pair (e.g. image and label).
#         print(self.labels[index])
        return (rgb_image, self.labels[index])
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)
