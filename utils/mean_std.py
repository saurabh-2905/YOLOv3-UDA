import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import tqdm

def calculate_ms(list_of_imgpaths):
    """
    Calculate mean and standard deviation over whole dataset

    list_of_imgpaths: list of paths to all images in dataset
    """
    img_path = list_of_imgpaths   ### load images

    tot_r = 0    ### store sum of r-channel
    tot_g = 0    ### store sum of g-channel
    tot_b = 0    ### store sum of b-channel
    tot_pixel = 0   ### store total number of pixels
    for i, img in enumerate(tqdm.tqdm(img_path, desc='Calculate mean')):
        ip_img = transforms.ToTensor()(Image.open(img).convert('RGB'))     ### read PIL image (H,W,C) and transform to tensor (C,H,W)
        h, w = ip_img.shape[1], ip_img.shape[2]    ### get height and width to calculate total number of pixels
        
        r_sum = ip_img[0].sum()  ### sum values of al pixels in single channel for one image
        g_sum = ip_img[1].sum()
        b_sum = ip_img[2].sum()
        num_pix = w * h
        
        tot_r += r_sum   ### sum the pixel values of all images for each channel
        tot_g += g_sum
        tot_b += b_sum
        tot_pixel += num_pix   ### calculate total number of pixel for all images in dataset
        
        #print(tot_pixel)
        
        # if i == 100:
        #     break

    r_mean = tot_r / tot_pixel   # calculate mean for each channel over whole dataset
    g_mean = tot_g / tot_pixel
    b_mean = tot_b / tot_pixel



    tot_r = 0     # reset variables for standard deviation
    tot_g = 0
    tot_b = 0

    for i, img in enumerate(tqdm.tqdm(img_path, desc='Calculate std')):
        ip_img = transforms.ToTensor()(Image.open(img).convert('RGB'))
        w, h = ip_img.shape[1], ip_img.shape[2]
        
        r_sr = (ip_img[0] - r_mean).pow(2).sum()   ### calculate sum of squared error for each channel in single image
        g_sr = (ip_img[1] - g_mean).pow(2).sum()
        b_sr = (ip_img[2] - b_mean).pow(2).sum()
        
        tot_r += r_sr   # add sum of SE of all images
        tot_g += g_sr
        tot_b += b_sr
        
        # if i == 100:
        #     break
            
    r_std = torch.sqrt(tot_r / tot_pixel)    ### divide SE by total number of pixels and take square root to get std
    g_std = torch.sqrt(tot_g / tot_pixel)
    b_std = torch.sqrt(tot_b / tot_pixel)

    print('Mean:', r_mean, g_mean, b_mean)
    print('Std. Deviation:', r_std, g_std, b_std)

    return [r_mean, g_mean, b_mean], [r_std, g_std, b_std]
