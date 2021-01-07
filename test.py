from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, json_path, iou_thres, conf_thres, nms_thres, img_size, batch_size, class_80, gpu_num, train_data= None):
    model.eval()

    # # Get dataloader
    # dataset = ImageAnnotation(folder_path=path, json_path=json_path, img_size=img_size, augment=False, multiscale=False, class_80=class_80)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    # )

    dataset = ListDataset(path, augment=False, multiscale=False, normalized_labels=False, pixel_norm=True, train_data=train_data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # img_paths = []  # Stores image paths
    # img_detections = []  # Stores detections for each image index
    val_acc_epoch = 0
    val_loss_epoch = 0
    for batch_i, (path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

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
            loss, outputs = model(imgs,in_targets)
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
        # # Save image paths and detections
        # img_paths.extend(path)
        # img_detections.extend(outputs)

        # if batch_i == 0:
        #         break

    # Calculat validation loss and accuracy
    val_acc_epoch = val_acc_epoch / (batch_i+1)
    val_loss_epoch = val_loss_epoch / (batch_i+1)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, val_acc_epoch, val_loss_epoch    #, img_paths[:20], img_detections[:20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/all_images/36_e3.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/class.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    gpu_no = 5
    device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    valid_annpath = data_config["json_val"]
    class_names = load_classes(data_config["names"])

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

    precision, recall, AP, f1, ap_class, val_acc, val_loss, = evaluate(
        model,
        path=valid_path,
        json_path=valid_annpath,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        class_80=class_80,
        gpu_num=device.index
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}",
            f"val_acc: {val_acc}",
            f"val_loss: {val_loss}")
