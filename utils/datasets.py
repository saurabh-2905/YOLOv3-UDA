import glob
import random
import os
import json
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from collections import defaultdict


from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
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

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ImageAnnotation(Dataset):
    def __init__(self, folder_path, json_path, multiscale=True, img_size=416, augment=True, normalized_labels=False):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.json_path = json_path
        self.normalized_labels = normalized_labels
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.img_ids = []
        # self.imgid2info = dict()
        self.imgid2path = dict()
        self.imgid2anns = defaultdict(list)
        self.catids = []
        if isinstance(folder_path, str):
            assert isinstance(json_path, str)
            img_dir, json_path = [folder_path], [json_path]
        assert len(img_dir) == len(json_path)
        for imdir, jspath in zip(img_dir, json_path):
            self.load_anns(imdir, jspath)


    def load_anns(self, img_dir, json_path):
        self.coco = False
        print(f'Loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        for ann in json_data['annotations']:
            img_id = ann['image_id']
            new_ann = None
            # get width and height 
            if not 'rbbox' in ann:
                # using COCO dataset. 4 = [x1,y1,w,h]
                self.coco = True
                # convert COCO format: x1,y1,w,h to x,y,w,h
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
                ann['bbox'].append(0)
                if ann['bbox'][2] > ann['bbox'][3]:
                    ann['bbox'][2], ann['bbox'][3] = ann['bbox'][3], ann['bbox'][2]
                    ann['bbox'][4] -= 90
                new_ann = ann['bbox']
            else:
                # using rotated bounding box datasets. 5 = [cx,cy,w,h,angle]
                # x,y,w,h,a
                assert len(ann['rbbox']) == 5, 'Unknown bbox format'
                new_ann = ann['rbbox']

                if new_ann[2] > new_ann[3]:
                    new_ann[2], new_ann[3] = new_ann[3], new_ann[2]
                    new_ann[4] += 90 - np.finfo(np.float32).eps

            if new_ann[2] == new_ann[3]:
                new_ann[3] += 1  # force that w < h

            assert new_ann[2] < new_ann[3]
            assert new_ann[4] >= -90 and new_ann[4] < 90

            # override original bounding box with rotated bounding box
            ann['bbox'] = torch.Tensor(new_ann)
            self.imgid2anns[img_id].append(ann)

        for img in json_data['images']:
            img_id = img['id']
            assert img_id not in self.imgid2path
            anns = self.imgid2anns[img_id]
            # if there is crowd gt, skip this image
            if self.coco and any(ann['iscrowd'] for ann in anns):
                continue

            self.img_ids.append(img_id)
            self.imgid2path[img_id] = os.path.join(img_dir, img['file_name'])
            # self.imgid2info[img['id']] = img

        self.catids = [cat['id'] for cat in json_data['categories']]

    def __getitem__(self, index):
        # -------
        # get image
        # -------
        img_id = self.img_ids[index]
        img_path = self.imgid2path[img_id]

        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape
        # -------
        # get label
        # -------
        targets = None
        if os.path.exists(self.json_path):
             # load unnormalized annotation
            annotations = self.imgid2anns[img_id]
            gt_num = len(annotations)
            boxes = torch.zero(gt_num,6)

            for i, ann in enumerate(annotations):
                boxes[i,1:] = ann['bbox']
                boxes[i,0] = self.catids.index(ann['category_id'])

            
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 7))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.files)
