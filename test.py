from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6 '  # 0,1,2,3,4,5,6
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


def evaluate(model, path, json_path, iou_thres, conf_thres, nms_thres, img_size, batch_size, class_80, gpu_num, use_angle, class_num, train_data= None):
    model.eval()

    # # Get dataloader
    # dataset = ImageAnnotation(folder_path=path, json_path=json_path, img_size=img_size, augment=False, multiscale=False, class_80=class_80)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    # )

    dataset = ListDataset(path, augment=False, multiscale=False, normalized_labels=False, pixel_norm=True, train_data=train_data, use_angle=use_angle, class_num=class_num)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
            loss, outputs = model(imgs, targets=in_targets, use_angle=use_angle)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres, use_angle=use_angle)

        val_acc_batch = 0
        for j, yolo in enumerate(model.yolo_layers):
            for name, metric in yolo.metrics.items():
                if name == "cls_acc":
                    val_acc_batch += metric

        # Accumulate loss for every batch of epoch
        val_acc_epoch += val_acc_batch / 3
        val_loss_epoch += loss.item()

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres, use_angle=use_angle)
        # # Save image paths and detections
        # img_paths.extend(path)
        # img_detections.extend(outputs)

        # if batch_i == 19:
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
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-rot-c6.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/testing.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="checkpoints/dst-fes/fda3norm_opt.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/class.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_angle", default=False, help='set flag to train using angle')
    #parser.add_argument('--train_dataset', type=str, default='dst', help='dataset on which model was trained')
    opt = parser.parse_args()
    print(opt)

    gpu_no = 0
    device = torch.device(f"cuda:{gpu_no}" if torch.cuda.is_available() else "cpu")
    if device.type != 'cpu':
        torch.cuda.set_device(device.index)
    print(device)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    valid_annpath = data_config["json_val"]
    class_names = load_classes(data_config["names"])

    if train_path.find('custom') != -1:   ### flag to use same mean and std values for evaluation as well
        train_dataset = 'theodore'
        print('Testing on Theodore Dataset')
    elif train_path.find('fes') != -1:
        train_dataset = 'fes'
        print('Testing on FES dataset')
    elif train_path.find('DST') != -1:
        train_dataset = 'dst'
        print('Testing on DST dataset')
    elif train_path.find('coco') != -1:
        train_dataset = 'coco'
        print('Training on COCO dataset')
    elif train_path.find('cepdof') != -1:
        train_dataset = 'cepdof_light'
        print('Training on CEPDOF dataset')
    else:
        raise FileNotFoundError('Invalid Dataset')


   # train_dataset = opt.train_dataset
    class_count = len(class_names)
    if len(class_names) == 80:
        class_80 = True
    else:
        class_80 = False
        

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    ### Load checkpoints
    checkpoint = torch.load(opt.pretrained_weights, map_location=lambda storage, loc:storage)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            if opt.pretrained_weights.find('opt') != -1:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_darknet_weights(opt.pretrained_weights)

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
        gpu_num=device.index,
        train_data=train_dataset,
        use_angle=opt.use_angle,
        class_num = class_count
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}",
            f"val_acc: {val_acc}",
            f"val_loss: {val_loss}")
