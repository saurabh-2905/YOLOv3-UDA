import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import glob
from utils.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
from utils.fda import adapt_images
import tqdm
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ' ' #0,1,2,3,4,5,6

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fda_type", type=str, default='np', choices=['np', 'normal'], help="select if images will be processed using torch of numpy" )
    parser.add_argument("--beta", type=float, default=0.01, choices=[0.01, 0.05, 0.005], help="factor to select size of mask. Should be between 0 and 1" )
    parser.add_argument("--circle_mask", type=bool, default=False, help="to select the circular mask. Default mask is square")
    opt = parser.parse_args()
    print(opt)

    # gpu_no = 6
    # device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")
    # if device.type != 'cpu':
    #     torch.cuda.set_device(device.index)
    # print(device)

    srcdataset_path = '/localdata/saurabh/yolov3/data/custom/images/person'
    trgdataset_path = '/localdata/saurabh/yolov3/data/cepdof/all_images'

    src_data = ImageFolder(srcdataset_path, img_size=416, augment=False )
    trg_data = ImageFolder(trgdataset_path, img_size=416, augment=False )

    src_loader = torch.utils.data.DataLoader(
            src_data, 
            batch_size=1,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    trg_loader = torch.utils.data.DataLoader(
            trg_data, 
            batch_size=1,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    trg_iter = enumerate(cycle(trg_loader))

    for batch_i, (src_imgpath, _) in enumerate( tqdm.tqdm(src_loader, desc='Fourier Adaptation' ) ):

        _, batch_uda = trg_iter.__next__()
        trg_imgpath, _ = batch_uda

        mixed, src_img, trg_img = adapt_images(src_imgpath[0], trg_imgpath[0], opt.fda_type, opt.beta, use_circular=opt.circle_mask,)

        out_dir = f'/localdata/saurabh/yolov3/data/fda/{opt.beta}/{os.path.basename(src_imgpath[0])}'

        if opt.fda_type == 'normal':
            save_image(mixed[0], out_dir)
            
            # if batch_i == 10:
            #     break

        else:
            mixed = (np.clip(mixed, 0, 1) * 255).astype(np.uint8)   ### giet in the format to save

            # out_dir = src_imgpath[0].replace('person', 'fda')
            mixed = Image.fromarray(mixed)
            # mixed.save(out_dir)
            # print(f'Saved output {out_dir}')
            

            # if batch_i == 10:
            #     break


