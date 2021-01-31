from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = ''  ## 0,1,2,3,4,5,6

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, class_80, gpu_num, train_data= None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False, normalized_labels=False, pixel_norm=True, train_data=train_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    val_acc_epoch = 0
    val_loss_epoch = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        if targets is None:
            continue

        in_targets = targets.detach().clone()
        in_targets = in_targets.to(device)

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
        targets[:, 2:6] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            loss, outputs = model(imgs, in_targets)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        val_acc_batch = 0
        for j, yolo in enumerate(model.yolo_layers):
            for name, metric in yolo.metrics.items():
                if name == "cls_acc":
                    val_acc_batch += metric

        # Accumulate loss for every batch of epoch
        val_acc_epoch += val_acc_batch / 3
        val_loss_epoch += loss.item()

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    val_acc_epoch = val_acc_epoch / (batch_i+1)
    val_loss_epoch = val_loss_epoch / (batch_i+1)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, val_acc_epoch, val_loss_epoch 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-standard-c1.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/testing.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/dst-fes/baseline1_theo.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/person.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    gpu_no = 5
    device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")
    print(device)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    valid_annpath = data_config["json_val"]
    class_names = load_classes(data_config["names"])

    print(f'Testing Dataset: {valid_path}')
    if train_path.find('custom') != -1:   ### flag to use same mean and std values for evaluation as well
        train_dataset = 'theodore'
        print('Normalize on Theodore Dataset')
    elif train_path.find('fes') != -1:
        train_dataset = 'fes'
        print('Normalize on FES dataset')
    elif train_path.find('DST') != -1:
        train_dataset = 'dst'
        print('Normalize on DST dataset')
    elif train_path.find('cepdof') != -1:
        train_dataset = 'cepdof'
        print('Normalize on DST dataset')

    if len(class_names) == 80:
        class_80 = True
    else:
        class_80 = False

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, val_acc, val_loss = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        class_80=class_80,
        gpu_num=device.index,
        train_data=train_dataset
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
