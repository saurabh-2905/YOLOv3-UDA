import glob
import random
import os
import json
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn.functional as F
from collections import defaultdict
from utils.utils import load_ms, write_ms
from utils.mean_std import calculate_ms
import glob
import warnings
import matplotlib.pyplot as plt

from utils.transforms import *
from utils.augmentations import DefaultAug
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.fda import FDA_source_to_target_np


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, train_data=None, img_size=416, augment=False ):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.pixel_norm = False
        self.augment = augment

        if train_data == 'theodore': 
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('/localdata/saurabh/yolov3/data/theodore_ms.txt')  ## use this if file missing [0.2731, 0.2540, 0.2319], [0.2328, 0.2240, 0.2125]
        elif train_data == 'fes':
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('data/fes/fes_ms.txt')
        elif train_data == 'dst':
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('data/DST/dst_ms.txt')
        elif train_data == 'cepdof':
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('data/cepdof/cepdof_ms.txt')
        elif train_data == 'theo_cep':
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('data/theo_cep_ms.txt')
        elif train_data == 'imagenet':
            self.pixel_norm = True
            self.mean_t, self.std_t = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif train_data == None:
            self.pixel_norm = False
        else:
            print('Using default values of Theodore for Normalization')
            self.pixel_norm = True
            self.mean_t, self.std_t = [0.2731, 0.2540, 0.2319], [0.2328, 0.2240, 0.2125]   ### Load values for theodore dataset as default 
            # raise RuntimeError('Invalid dataset')


    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = np.array(Image.open(img_path).convert('RGB'), dtype='uint8')
        boxes = np.zeros((1, 6))

        if self.augment == True:
            # Label Placeholder
            tran = transforms.Compose([
                DefaultAug(),
                PadSquare(),
                RelativeLabels(),
                ToTensor(),
                ])
        else:
            tran = transforms.Compose([
                PadSquare(),
                RelativeLabels(),
                ToTensor(),
                ])
        
        img, _ = tran((img,boxes))

        if self.pixel_norm == True:
            img = transforms.Normalize(self.mean_t, self.std_t)(img)

        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, use_angle, class_num, img_size=416, augment=True, multiscale=True, normalized_labels=True,
                     pixel_norm=False, train_data=None, uda_method=None, beta=0.01, circular=False ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        '''if uda_method == 'fda':
            self.trg_files = glob.glob('/localdata/saurabh/yolov3/data/cepdof/all_images/*')'''

        if use_angle == 'True':
            if class_num == 1:
                self.label_files = [
                    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                    for path in self.img_files
                ]
            elif class_num == 6:
                self.label_files = [
                    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt").replace('person', 'all_class')
                    for path in self.img_files
                ]
        else:
            if class_num == 1:
                self.label_files = [
                    path.replace("images", "labelsbbox").replace(".png", ".txt").replace(".jpg", ".txt")        #.replace('fda', 'person')
                    for path in self.img_files
                ]
            elif class_num == 6:
                self.label_files = [
                    path.replace("images", "labelsbbox").replace(".png", ".txt").replace(".jpg", ".txt").replace('person', 'all_class')           #.replace('fda', 'all_class')
                    for path in self.img_files
                ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.pixel_norm = pixel_norm
        self.uda_method = uda_method
        self.beta = beta
        self.circular = circular

        if use_angle == True:
            self.augment = False
        
        if self.pixel_norm == True:
            print(f'Training on Normalized Pixels of {train_data}')
            if train_data == 'theodore': 
                self.mean_t, self.std_t = load_ms('/localdata/saurabh/yolov3/data/theodore_ms.txt')

            elif train_data == 'fes': 
                mean_path = 'data/fes/fes_ms.txt' 
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    fes_imgpath = glob.glob('/localdata/saurabh/dataset/FES/JPEGImages/*.jpg')
                    self.mean_t, self.std_t = calculate_ms(fes_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )
            
            elif train_data == 'dst':
                mean_path = 'data/DST/dst_ms.txt' 
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    fes_imgpath = glob.glob('/localdata/saurabh/dataset/DST/val/*.png')
                    self.mean_t, self.std_t = calculate_ms(fes_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )

            elif train_data == 'coco':
                mean_path = 'data/coco/coco_ms.txt'
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    coco_imgpath = glob.glob('/localdata/saurabh/yolov3/data/coco/images/train2017/*.jpg')
                    self.mean_t, self.std_t = calculate_ms(coco_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )

            elif train_data == 'cepdof_light':
                mean_path = 'data/cepdof/cepdof_ms.txt'
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    cepdof_imgpath = glob.glob('/localdata/saurabh/yolov3/data/cepdof/images/person/*')
                    self.mean_t, self.std_t = calculate_ms(cepdof_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )

            elif train_data == 'mwr':
                mean_path = 'data/mwr/mwr_ms.txt'
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    mwr_imgpath = glob.glob('/localdata/saurabh/yolov3/data/mwr/images/person/*')
                    self.mean_t, self.std_t = calculate_ms(mwr_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )

            elif train_data == 'theo_cep':
                mean_path = 'data/theo_cep_ms.txt'
                if os.path.isfile( mean_path ) == True:
                    self.mean_t, self.std_t = load_ms(mean_path)
                else:
                    cepdof_imgpath = glob.glob('/localdata/saurabh/yolov3/data/cepdof/all_images/*')
                    custom_imgpath = glob.glob('/localdata/saurabh/yolov3/data/custom/images/person/*')
                    all_imgpath = cepdof_imgpath + custom_imgpath
                    self.mean_t, self.std_t = calculate_ms(all_imgpath)
                    mean_std = [self.mean_t, self.std_t]
                    write_ms( mean_path, mean_std )

            elif train_data == 'imagenet':
                self.pixel_norm = True
                self.mean_t, self.std_t = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def __getitem__(self, index):
        #---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        #print(img_path)
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8) #/255.0

        '''if self.uda_method == 'fda':
            trg_path = self.trg_files[ np.random.randint(len(self.trg_files)) ]
            trg_img = Image.open(trg_path).convert('RGB')
            trg_img = trg_img.resize(img.shape[:2], resample=Image.BILINEAR)
            # trg_img.save('sample.png')
            trg_img = np.array(trg_img, dtype=np.uint8) #/255.0
            
            img= img.transpose(2,0,1)
            trg_img= trg_img.transpose(2,0,1)

            img = FDA_source_to_target_np(img, trg_img, L=self.beta, use_circular=self.circular)   ### expect images in (c,h,w)
            ########### not in fda exp #############
            # img = (255*(img - np.min(img))/np.ptp(img))     #### to normalize post. and neg. values in range (0,255)
            img = img.clip( 0,255)
            ########################################
            img = img.transpose(1,2,0).astype(np.uint8)
            
            ########### for testing ##################
            # img = (np.clip(img, 0, 255)).astype(np.uint8)   ### giet in the format to save
            # plt.imshow(img)
            # plt.savefig(f'fda_samples/{self.circular}_{self.beta}_{os.path.basename(img_path)}')
            ##########################################'''


        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 6)

            if self.normalized_labels == True: 
                w, h, _ = img.shape 
                boxes[:,[1,3]] *= h
                boxes[:,[2,4]] *= w

            if self.augment == True:
                tran = transforms.Compose([
                DefaultAug(),
                PadSquare(),
                RelativeLabels(),
                ToTensor(),
                ])
            else:
                tran = transforms.Compose([
                PadSquare(),
                RelativeLabels(),
                ToTensor(),
                ])

            img, targets = tran((img,boxes))

        if img.shape[2] == 3:
            img = transforms.ToTensor()(img)

        if self.pixel_norm == True:
            img = transforms.Normalize(self.mean_t, self.std_t)(img) 

        return img_path, img, targets

    
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            if boxes is None:
                continue
            boxes[:, 0] = i
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        #targets = torch.cat(targets, 0)

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError as e_inst:
            targets = None # No boxes for an image
            
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

