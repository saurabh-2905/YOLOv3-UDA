import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def FDA_source_to_target(src_img, trg_img, L=0.1, use_circular=False):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft(
        src_img.clone().to(
            src_img.device),
        signal_ndim=2,
        onesided=False)
    fft_trg = torch.rfft(
        trg_img.clone().to(
            src_img.device),
        signal_ndim=2,
        onesided=False)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(
        amp_src.clone().to(
            src_img.device), amp_trg.clone().to(
            src_img.device), L=L, use_circular=use_circular)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_[:, :, :, :, 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:, :, :, :, 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft(
        fft_src_.to(src_img.device),
        signal_ndim=2,
        onesided=False,
        signal_sizes=[
            imgH,
            imgW])

    return src_in_trg


def FDA_source_to_target_np(src_img, trg_img, L=0.1, use_circular=False):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img.cpu().numpy()
    trg_img_np = trg_img.cpu().numpy()

    # src_img_np = np.uint8(src_img_np * 255)
    # trg_img_np = np.uint8(trg_img_np * 255)

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L, use_circular=use_circular)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:, :, :, :, 0]**2 + fft_im[:, :, :, :, 1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1, use_circular=False):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)     # get b

    if use_circular:
        axes = (int(h * L), int(w * L))
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask = cv2.ellipse(mask, (0, 0), axes, 0, 0, 360,
                           (255, 255, 255), -1).astype(np.bool)
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).to(amp_src.device)
        amp_src = amp_src * mask + amp_trg * ~mask
    else:

        amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
        amp_src[:, :, 0:b, w - b:w] = amp_trg[:,
                                              :, 0:b, w - b:w]    # top right
        amp_src[:, :, h - b:h, 0:b] = amp_trg[:,
                                              :, h - b:h, 0:b]    # bottom left
        amp_src[:, :, h - b:h, w - b:w] = amp_trg[:,
                                                  :, h - b:h, w - b:w]  # bottom right
    return amp_src


def low_freq_mutate_np(amp_src, amp_trg, L=0.1, use_circular=False):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    if use_circular:
        axes = (int(h * L), int(w * L))
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask = cv2.ellipse(mask, (c_w, c_h), axes, 0, 0, 360,
                           (255, 255, 255), -1).astype(np.bool)
        mask = mask.transpose(2, 0, 1)
        a_src = a_trg * mask + a_src * ~mask

    else:
        h1 = c_h - b
        h2 = c_h + b #+ 1
        w1 = c_w - b
        w2 = c_w + b #+ 1

        a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]

    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def adapt_images(src_path, trg_path, fda_type, beta, use_circular=False, ):
    '''
    Arguments
    src_path: path to source image (str)
    trg_path: path to target image (str)
    fda_type: process images as numpy or tensor (str) ['np', 'normal']
    beta: factor by which low level features must be replaced (int)
    use_circular: to use circular mask (bool)
    
    Returns
    mixed: the fourier adapted image (w,h,c)
    src_img: source image
    trg_img: target image
    '''
    # gpu_no = 6
    # device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")
    # if device.type != 'cpu':
    #     torch.cuda.set_device(device.index)
    # # print(device)

    src_img = transforms.ToTensor()(Image.open(src_path).convert('RGB'))
    trg_img = transforms.ToTensor()(Image.open(trg_path).convert('RGB'))
    #### resize images to the size of netwrok input 
    trg_img = resize(trg_img, (src_img.shape[1:]) ) 



    if fda_type == 'np':
        # print('FDA using numpy')
        mixed = FDA_source_to_target_np(src_img, trg_img, L=beta, use_circular=use_circular)
        
        src_img = src_img.permute(1,2,0).contiguous()
        trg_img = trg_img.permute(1,2,0).contiguous()
        mixed = mixed.transpose(1,2,0)
        
        # mixed += abs(mixed.min())
        
    elif fda_type == 'normal':
        # print('FDA using tensor')
        src_img = torch.unsqueeze(src_img, dim=0) #.to(device)
        trg_img = torch.unsqueeze(trg_img, dim=0) #.to(device)

        mixed = FDA_source_to_target(src_img, trg_img, L=beta, use_circular=use_circular)
        
        # src_img = src_img[0].permute(1,2,0).contiguous()
        # trg_img = trg_img[0].permute(1,2,0).contiguous()
        # mixed = mixed[0].permute(1,2,0).contiguous()

    return mixed, src_img, trg_img

if __name__=='__main__':
    fda_type = 'np' #[np, normal]
    beta = 0.01
    circle_mask = False

    src_path = '/localdata/saurabh/yolov3/data/custom/images/person/0026118_img.png'
    trg_path =  '/localdata/saurabh/yolov3/data/fes/images/person/Record_00773.jpg'

    mixed, src_img, trg_img = adapt_images(src_path, trg_path, fda_type, beta, use_circular=circle_mask)
    mixed = (np.clip(mixed, 0, 1) * 255).astype(np.uint8)

    # mixed = (255.0 / mixed.max() * (mixed - mixed.min())).astype(np.uint8)
    mixed = Image.fromarray(mixed)
    mixed.save('trial_fda.png')

    
    


