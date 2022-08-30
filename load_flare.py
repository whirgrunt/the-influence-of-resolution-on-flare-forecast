import os
import torch
import numpy as np
import scipy.misc as m
import imageio
import random
random.seed(1234)
from torch.utils import data
from torchvision import transforms
import torchvision
import cv2
from PIL import Image
import random
root_path = './data/magnetogram_jpg/'
#file_path = '/exdata/lsx/dataset/'
class flare(data.Dataset):

    def __init__(
        self,
        reduce_resolution,
        cross,
        cross_test_year,
        split="train",
        img_size=(512, 512),
        img_norm=True,
        transform=None,
    ):

        if os.path.exists(root_path):
            self.root = root_path
        else:
            raise ValueError('data does not exist!')
        self.split = split
        self.img_norm = img_norm
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.reduce_resolution = reduce_resolution
        self.files = {}
        self.path_read = []
        if cross == False:
            if split == 'train':
                with open('./train_neg.txt', "r") as f1:
                    self.files_neg = f1.read().splitlines()
                    self.neg_num = len(self.files_neg)
                with open( './train_pos.txt', "r") as f:
                    self.files[split] = f.read().splitlines()
            elif split == 'test':
                with open( './test.txt', "r") as f:
                    self.files[split] = f.read().splitlines()
            else:
                print("image split is error!")
        if cross == True:
            if split == 'train':
                with open('./txt_cross/' + str(cross_test_year) + '_train_neg.txt', "r") as f1:
                    self.files_neg = f1.read().splitlines()
                    self.neg_num = len(self.files_neg)
                with open( './txt_cross/' + str(cross_test_year) + '_train_pos.txt', "r") as f:
                    self.files[split] = f.read().splitlines()
            elif split == 'test':
                with open( './txt_cross/' + str(cross_test_year) + '_test.txt', "r") as f:
                    self.files[split] = f.read().splitlines()
            else:
                print("image split is error!")

        #self.files[split].sort()
        self.transform = transform

        if not self.files[split]:
            raise Exception("No files for split=[%s] found" % (split))
        print("\nFound %d %s images" % (len(self.files[split]), split))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_info = self.files[self.split][index].rstrip()
        img_path = img_info.split(' ')[0]
        img = Image.open(root_path + img_path + '.jpg').convert('RGB')
        gt = int(img_info.split(' ')[1])

        #img = transforms.ToPILImage(img)
        img = img.resize((max(img.size[0] // self.reduce_resolution,1), max(img.size[1] // self.reduce_resolution,1)))
        sample = self.transform(img)

        rand_neg = random.randint(0, self.neg_num - 1)
        img_info_neg = self.files_neg[rand_neg]
        img_path_neg = img_info_neg.split(' ')[0]
        img_neg = Image.open(root_path + img_path_neg + '.jpg').convert('RGB')
        gt_neg = int(img_info_neg.split(' ')[1])
        img_neg = img_neg.resize((max(img_neg.size[0] // self.reduce_resolution,1), max(img_neg.size[1] // self.reduce_resolution,1)))

        sample_neg = self.transform(img_neg)




        '''
        unloader = torchvision.transforms.ToPILImage()
        image = sample.cpu().clone()  # clone the tensor
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.save('example.jpg')
        '''
        return sample,gt, sample_neg, gt_neg

class flare_test(data.Dataset):

    def __init__(
        self,
        reduce_resolution,
        cross,
        cross_test_year,
        split="test",
        img_size=(512, 512),
        img_norm=True,
        transform=None,
    ):

        if os.path.exists(root_path):
            self.root = root_path
        else:
            raise ValueError('data does not exist!')
        self.split = 'test'
        self.img_norm = img_norm
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}
        self.path_read = []
        self.reduce_resolution = reduce_resolution
        if cross == False:
            with open( './test.txt', "r") as f:
                self.files[split] = f.read().splitlines()
        if cross == True:
            with open( './txt_cross/' + str(cross_test_year) + '_test.txt', "r") as f:
                self.files[split] = f.read().splitlines()

        #self.files[split].sort()
        self.transform = transform

        if not self.files[split]:
            raise Exception("No files for split=[%s] found" % (split))
        print("\nFound %d %s images" % (len(self.files[split]), split))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_info = self.files[self.split][index].rstrip()
        img_path = img_info.split(' ')[0]
        img = Image.open(root_path + img_path + '.jpg').convert('RGB')
        gt = int(img_info.split(' ')[1])
        img = img.resize((max(img.size[0] // self.reduce_resolution,1), max(img.size[1] // self.reduce_resolution,1)))
        sample = self.transform(img)

        return sample, gt