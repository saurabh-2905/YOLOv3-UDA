class ImageAnnotation(Dataset):
    def __init__(self, folder_path, json_path, img_size=416, multiscale=True,  augment=True, normalized_labels=False, class_80=False):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        #self.files = [files for files in self.files if files]
        self.img_size = img_size
        #self.img_norm_bbox = 1280
        self.json_path = json_path
        self.normalized_labels = normalized_labels
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.yolo = class_80

        self.label_map = dict()
        self.names = []
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
        #self.label_mapping()
        self.pixel_norm = False

        if self.json_path[0].find('custom') != -1 or self.json_path[0].find('theodore') != -1:
            self.pixel_norm = True
            self.mean_t, self.std_t = load_ms('/localdata/saurabh/yolov3/data/theodore_ms.txt')

        elif self.json_path[0].find('fes') != -1:
            self.pixel_norm = True
            mean_path = os.path.join( '/localdata/saurabh/yolov3/data/fes/', 'fes_ms.txt' )
            if os.path.isfile( mean_path ) == True:
                self.mean_t, self.std_t = load_ms(mean_path)
            else:
                fes_imgpath = glob.glob('/localdata/saurabh/dataset/FES/JPEGImages/*.jpg')
                self.mean_t, self.std_t = calculate_ms(fes_imgpath)
                mean_std = [self.mean_t, self.std_t]
                write_ms( mean_path, mean_std )
        
        elif self.json_path[0].find('DST') != -1:
            self.pixel_norm = True
            mean_path = os.path.join( '/localdata/saurabh/yolov3/data/dst/', 'dst_ms.txt' )
            if os.path.isfile( mean_path ) == True:
                self.mean_t, self.std_t = load_ms(mean_path)
            else:
                fes_imgpath = glob.glob('/localdata/saurabh/dataset/DST/val/*.png')
                self.mean_t, self.std_t = calculate_ms(fes_imgpath)
                mean_std = [self.mean_t, self.std_t]
                write_ms( mean_path, mean_std )

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
                if not self.yolo:
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
                    new_ann[4] -= 90 if new_ann[4] > 0 else -90

            if new_ann[2] == new_ann[3]:
                new_ann[3] += 1  # force that w < h

            new_ann[4] = np.clip(new_ann[4], -90.0, 90.0 - 1e-14)

            if not self.yolo:
                assert new_ann[2] < new_ann[3]
                assert new_ann[4] >= -90 and new_ann[4] < 90

            # Normalize the bbox coordinates between 0 to 1
            # new_ann[0] /= self.img_norm_bbox
            # new_ann[1] /= self.img_norm_bbox
            # new_ann[2] /= self.img_norm_bbox
            # new_ann[3] /= self.img_norm_bbox
            # override original bounding box with rotated bounding box
            ann['bbox'] = torch.Tensor(new_ann)
            self.imgid2anns[img_id].append(ann)

        for img in json_data['images']:
            img_id = img['id']
            assert img_id not in self.imgid2path

            self.img_ids.append(img_id)
            self.imgid2path[img_id] = os.path.join(img_dir, img['file_name'])
            # self.imgid2info[img['id']] = img

        self.catids = [cat['id'] for cat in json_data['categories']]
        self.names = [cat['name'] for cat in json_data['categories']]

    def label_mapping(self):
        for i in range(6):
            self.label_map[i+1] = i 

    def __getitem__(self, index):
        # -------
        # get image
        # -------
        self.index = index % len(self.imgid2path)
        img_id = self.img_ids[self.index]
        img_path = self.imgid2path[img_id]
        #print(img_path)

        if self.pixel_norm == True:
            img = (Image.open(img_path).convert('RGB'))
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean_t, self.std_t)
                ])
            img = trans(img)

        else:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            
        #print(img.shape)
        #Resize image to fixed size
        #img = resize(img, 416)
        #print(img.shape)

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
            boxes = torch.zeros(gt_num,6)

            for i, ann in enumerate(annotations):
                # Normalize the bbox coordinates between 0 to 1
                # ann['bbox'][0] /= self.img_norm_bbox
                # ann['bbox'][1] /= self.img_norm_bbox
                # ann['bbox'][2] /= self.img_norm_bbox
                # ann['bbox'][3] /= self.img_norm_bbox
                #print(ann['bbox'])
                boxes[i,1:] = ann['bbox']
                #print(boxes[i,1:])
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

            for box_batch in boxes:
                assert torch.isinf(box_batch[0]) == False
                assert torch.isinf(box_batch[1]) == False
                assert torch.isinf(box_batch[2]) == False
                assert torch.isinf(box_batch[3]) == False 

            targets = torch.zeros((len(boxes), 7))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

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
        return len(self.files)


#########################################
#### datasets.py class ListFolder ######
##########################################

# def __getitem__(self, index):

    #     # ---------
    #     #  Image
    #     # ---------

    #     img_path = self.img_files[index % len(self.img_files)].rstrip()
    #     #print(img_path)

    #     # Extract image as PyTorch tensor
    #     if self.pixel_norm == True:
    #         img = (Image.open(img_path).convert('RGB'))
    #         trans = transforms.Compose([
    #             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    #             transforms.ToTensor(),
    #             transforms.Normalize(self.mean_t, self.std_t)
    #             ])
    #         img = trans(img)

    #     else:
    #         img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

    #     # Handle images with less than three channels
    #     if len(img.shape) != 3:
    #         img = img.unsqueeze(0)
    #         img = img.expand((3, img.shape[1:]))

    #     _, h, w = img.shape
    #     h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
    #     # Pad to square resolution
    #     img, pad = pad_to_square(img, 0)
    #     _, padded_h, padded_w = img.shape

    #     # ---------
    #     #  Label
    #     # ---------

    #     label_path = self.label_files[index % len(self.img_files)].rstrip()

    #     targets = None
    #     if os.path.exists(label_path):
    #         # Ignore warning if file is empty
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
    #             # Extract coordinates for unpadded + unscaled image
    #             x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    #             y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    #             x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    #             y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
    #             # Adjust for added padding
    #             x1 += pad[0]
    #             y1 += pad[2]
    #             x2 += pad[1]
    #             y2 += pad[3]
    #             # Returns (x, y, w, h)
    #             boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    #             boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    #             boxes[:, 3] *= w_factor / padded_w
    #             boxes[:, 4] *= h_factor / padded_h

    #             targets = torch.zeros((len(boxes), 7))
    #             targets[:, 1:] = boxes

    #         # Apply augmentations
    #         if self.augment:
    #             if np.random.random() < 0.5:
    #                 img, targets = horisontal_flip(img, targets)

    #     return img_path, img, targets


####################################
######## augmentation.py ###########
#####################################

# import torch
# import torch.nn.functional as F
# import numpy as np

# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.polys import MultiPolygon, Polygon, PolygonsOnImage


# def horisontal_flip(images, targets):
#     images = torch.flip(images, [-1])
#     targets[:, 2] = 1 - targets[:, 2]
#     return images, targets

#########################################
#### datasets.py class ImageFolder ######
##########################################

def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = np.array(Image.open(img_path).convert('RGB'), dtype='uint8')
        boxes = np.zeros((1, 5))

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
        #     trans = transforms.Compose([
        #         transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        #         # transforms.RandomResizedCrop(self.img_size),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(self.mean_t, self.std_t)
        #         ])
        #     img = trans(img)

        # else:
        #     img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        
        # # Pad to square resolution
        # img, _ = pad_to_square(img, 0)
        # # Resize
        # img = resize(img, self.img_size)

        return img_path, img