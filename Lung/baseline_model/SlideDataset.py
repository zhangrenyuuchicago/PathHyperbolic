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
    def __init__(self, stage, lung_file, dir, transform, fold):
        '''
        read phenotypes
        '''
        self.stage = stage
        assert stage in {'train', 'val', 'test'}
        #self.inst_num = inst_num 
        
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
        
        print(f'lung type records: {len(self.slide_lung)}')
        ids = glob.glob(dir + "/*.jpg") 
        print(f'tiles num: {len(ids)}')

        self.slide_img = {}
        for img_name in ids:
            basename = ntpath.basename(img_name)
            array = basename.split("_")
            slide_id = array[0]
            if slide_id in self.slide_img:
                self.slide_img[slide_id].append(img_name)
            else:
                self.slide_img[slide_id] = [img_name]
        
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

        #print('number of each catag')
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
        
        img_name = random.choice(self.slide_img[slide_id])
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, lung, slide_id

    def __len__(self):
        return len(self.slide_id)


