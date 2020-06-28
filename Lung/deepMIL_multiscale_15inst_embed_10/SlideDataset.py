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

lung_index = {'NORM':0, 'LUAD':1, 'LUSC':2}

class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, stage, lung_file, dir_high, dir_med, dir_low, transform, fold, inst_num=10):
        '''
        read phenotypes
        '''
        self.stage = stage
        assert stage in {'train', 'val', 'test'}
        self.inst_num = inst_num 
        self.slide_lung = {}
        fin = open(lung_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            array = line.split()
            if array[1] not in lung_index:
                continue
            lung = lung_index[array[1]]
            slide_id = array[0]
            self.slide_lung[slide_id] = lung
        fin.close()

        ids_high = glob.glob(dir_high + "/*.jpg") 
        ids_med = glob.glob(dir_med + "/*.jpg") 
        ids_low = glob.glob(dir_low + "/*.jpg") 
        print(f'high tiles num: {len(ids_high)}')
        print(f'med tiles num: {len(ids_med)}')
        print(f'low tiles num: {len(ids_low)}')

        self.slide_img_high = {}
        self.slide_img_med = {}
        self.slide_img_low = {}
        
        # high
        for img_name in ids_high:
            basename = ntpath.basename(img_name)
            array = basename.split("_")
            slide_id = array[0]
            if slide_id in self.slide_img_high:
                self.slide_img_high[slide_id].append(img_name)
            else:
                self.slide_img_high[slide_id] = [img_name]
        
        # med
        for img_name in ids_med:
            basename = ntpath.basename(img_name)
            array = basename.split("_")
            slide_id = array[0]
            if slide_id in self.slide_img_med:
                self.slide_img_med[slide_id].append(img_name)
            else:
                self.slide_img_med[slide_id] = [img_name]
        
        # low
        for img_name in ids_low:
            basename = ntpath.basename(img_name)
            array = basename.split("_")
            slide_id = array[0]
            if slide_id in self.slide_img_low:
                self.slide_img_low[slide_id].append(img_name)
            else:
                self.slide_img_low[slide_id] = [img_name]

        fold_slide_lt = None
        with open('../settings/fold_slide_lt.json') as f:
            fold_slide_lt = json.load(f)

        if stage == 'train':
            fold_set = []
            for i in range(3):
                fold_i = (fold + i) % 5
                fold_set.append(fold_i)
            mask_id = []
            for fold_i in fold_set:
                mask_id += fold_slide_lt[fold_i]
        elif stage == 'val':
            assert fold < 5
            fold_i = (fold + 3) % 5
            mask_id = fold_slide_lt[fold_i]
        else:
            assert fold < 5
            fold_i = (fold + 4) % 5
            mask_id = fold_slide_lt[fold_i]

        self.slide_id = []
        for slide_id in mask_id:
            if slide_id in self.slide_lung:
                self.slide_id.append(slide_id)
        
        print(f'slides num: {len(self.slide_id)}')

        num_lung = {}
        for lung in set(self.slide_lung.values()):
            num_lung[lung] = 0

        for slide_id in self.slide_id:
            lung = self.slide_lung[slide_id]
            num_lung[lung] += 1

        #print(num_age_site_gender)
        if self.stage == 'train':
            weight_lt = []
            for slide_id in self.slide_id:
                lung = self.slide_lung[slide_id]
                weight_lt.append(1.0 / num_lung[lung])
            weight = np.array(weight_lt)
            np.savetxt('weight.txt', weight)
        
        self.transform = transform
        print( "Initialize end")

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        slide_id = self.slide_id[index]
        lung = self.slide_lung[slide_id]
        lung = torch.LongTensor([lung])
        
        image_lt = []
        sample_high_lt = random.sample(self.slide_img_high[slide_id], int(self.inst_num/3))
        sample_med_lt = random.sample(self.slide_img_med[slide_id], int(self.inst_num/3))
        sample_low_lt = random.sample(self.slide_img_low[slide_id], int(self.inst_num/3))
        sample_lt = sample_high_lt + sample_med_lt + sample_low_lt
        for img_name in sample_lt:
            image = Image.open(img_name)
            if self.transform is not None:
                image = self.transform(image)
                image_lt.append(image)

        image_lt = torch.stack(image_lt)
        return image_lt, lung, slide_id

    def __len__(self):
        return len(self.slide_id)


