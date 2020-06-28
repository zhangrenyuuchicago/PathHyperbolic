import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import sys
import glob, os
import ntpath
from torch.autograd import Variable 
import torchvision
import json
import random

class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, bag0_file, bag1_file, transform):
        self.bag0_file = bag0_file
        self.bag1_file = bag1_file
        
        self.bags = []
        self.bags_label = []

        fin = open(bag0_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            array = line.split(' ')
            self.bags.append(array)
            self.bags_label.append(0)
        fin.close()

        fin = open(bag1_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            array = line.split(' ')
            self.bags.append(array)
            self.bags_label.append(1)
        fin.close()
        
        self.transform = transform
        print( "Initialize end")

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        label = self.bags_label[index]
        label = torch.LongTensor([label])
        
        image_lt = []
        #sample_lt = random.sample(self.slide_img[slide_id], self.inst_num)
        sample_lt = self.bags[index]
        for img_name in sample_lt:
            image = Image.open(img_name)
            if self.transform is not None:
                image = self.transform(image)
                image_lt.append(image)
        image_lt = torch.stack(image_lt)
        return image_lt, label, str(index)

    def __len__(self):
        return len(self.bags)


