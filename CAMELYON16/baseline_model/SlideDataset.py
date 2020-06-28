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

label_index = {'Normal':0, 'Tumor':1}

class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, stage, label_file, dir_low, transform, fold):
        '''
        read phenotypes
        '''
        self.stage = stage
        assert stage in {'train', 'val', 'test'}
        self.slide_label = {}
        fin = open(label_file, 'r')
        while True:
            line = fin.readline().strip()
            if not line:
                break
            array = line.split()
            slide_id = array[0]
            label = array[1]
            if label not in label_index:
                continue
            label = label_index[label]
            self.slide_label[slide_id] = label
        fin.close()

        ids_low = glob.glob(dir_low + "/*/*.jpg") 
        print(f'low tiles num: {len(ids_low)}')

        self.slide_img_low = {}
        
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
        with open('../../settings/fold_slide_lt.json') as f:
            fold_slide_lt = json.load(f)

        if stage == 'train':
            fold_set = []
            for i in range(4):
                fold_i = (fold + i) % 5
                fold_set.append(fold_i)
            mask_id = []
            for fold_i in fold_set:
                mask_id += fold_slide_lt[fold_i]
        elif stage == 'val':
            assert fold < 5
            fold_i = (fold + 4) % 5
            mask_id = fold_slide_lt[fold_i]
        else:
            print('test, use all')            

        self.slide_id = []
        
        if stage == 'train' or stage == 'val':
            for slide_id in mask_id:
                self.slide_id.append(slide_id)
        else:
            for slide_id in self.slide_img_low:
                self.slide_id.append(slide_id)

        print(f'slides num: {len(self.slide_id)}')

        num_label = {}
        for label in set(self.slide_label.values()):
            num_label[label] = 0

        for slide_id in self.slide_id:
            label = self.slide_label[slide_id]
            num_label[label] += 1

        #print(num_age_site_gender)
        if self.stage == 'train':
            weight_lt = []
            for slide_id in self.slide_id:
                label = self.slide_label[slide_id]
                weight_lt.append(1.0 / num_label[label])
            weight = np.array(weight_lt)
            np.savetxt('weight.txt', weight)
        
        self.transform = transform
        print( "Initialize end")

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        slide_id = self.slide_id[index]
        label = self.slide_label[slide_id]
        label = torch.LongTensor([label])
        
        '''
        image_lt = []
        
        if len(self.slide_img_low[slide_id]) < self.inst_num:
            sample_low_lt = random.choices(self.slide_img_low[slide_id], k=self.inst_num)
        else:
            sample_low_lt = random.sample(self.slide_img_low[slide_id], self.inst_num)
        
        sample_lt =  sample_low_lt
        for img_name in sample_lt:
            image = Image.open(img_name)
            if self.transform is not None:
                image = self.transform(image)
                image_lt.append(image)

        image_lt = torch.stack(image_lt)
        '''
        img_name = random.choice(self.slide_img_low[slide_id])
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, slide_id

    def __len__(self):
        return len(self.slide_id)


