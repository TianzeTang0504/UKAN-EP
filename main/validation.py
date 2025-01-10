import argparse
import os
from collections import OrderedDict
import glob
import random
import numpy as np
import json
import torch.nn.functional as F
import results.unet
from results.aunet import AttentionUNet
import results.ukan
import results.ecapan
import results.ecapancnn
import results.ukanpan
import results.ukaneca
from results.swin_unetr import SwinUNETR

import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion
import h5py


import numpy as np
import os
import zipfile
import nibabel as nib
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import ukan
from scipy.ndimage import rotate


from kannet.utils import AverageMeter, str2bool


import shutil
import os
import subprocess

from pdb import set_trace as st


class Config:
    def __init__(self):
        self.num_classes = 5
        self.seed = 21
        self.epochs = 100
        self.warmup_epochs = 10
        self.batch_size = 1
        self.lr = 0.01
        self.min_lr = 0.005
        self.data_path = '/home/tianzetang/brats/data24/dataset/'  # train_set['out']
        #self.data_path = '/scratch/tt2631/brats24/data/dataset'  # train_set['out']
        self.train_txt = './train_split.txt'
        self.valid_txt = './test811.txt'
        # self.test_txt = '.\\test_split.txt'
        self.train_log = './output/UNet.txt'
        self.weights = './output/UNet.pth'
        self.save_path = './output'


args = Config()

random.seed(21)
np.random.seed(21)

if torch.cuda.is_available():
    #print("CUDA is available. GPU will be used for training.")
    device = torch.device("cuda")
else:
    #print("CUDA is not available. Training will be on CPU.")
    device = torch.device("cpu")


class RandomCrop(object):
    def __init__(self, output_size):
        assert len(output_size) == 3
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        _, d, h, w = image.shape
        new_d, new_h, new_w = self.output_size
        
        if d - new_d < 0 or h - new_h < 0 or w - new_w < 0:
            raise ValueError("Output size must be smaller than the dimensions of the input image")

        d1 = 13
        h1 = 11
        w1 = 11

        image = image[:, d1:d1 + new_d, h1:h1 + new_h, w1:w1 + new_w]
        label = label[d1:d1 + new_d, h1:h1 + new_h, w1:w1 + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}
    

class BraTS(Dataset):
    def __init__(self,data_path, file_path,transform=None):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        self.paths = [os.path.join(data_path, x + '-mri_norm2.h5') for x in lines]
        self.sample_names = lines
        self.transform = transform

    def __getitem__(self, item):
        sample_name = self.sample_names[item]
        h5f = h5py.File(self.paths[item], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label'], sample_name

    def __len__(self):
        return len(self.sample_names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v
    return new_state_dict


def dice_coefficient(preds, labels, num_classes):
    batch_size = preds.shape[0]
    dice_scores = torch.zeros(batch_size, num_classes, device=preds.device)

    # 单类别的 Dice 分数计算
    
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        label_cls = (labels == cls).float()

        intersection = (pred_cls * label_cls).sum()
        union = pred_cls.sum() + label_cls.sum()

        dice = 2.0 * intersection / union
        dice_scores[:, cls - 1] = dice

    # Whole Tumor Dice 分数计算 (由标签 1, 2, 3 组成)
    pred_whole_tumor = ((preds == 1) | (preds == 2) | (preds == 3)).float()
    label_whole_tumor = ((labels == 1) | (labels == 2) | (labels == 3)).float()

    intersection_whole = (pred_whole_tumor * label_whole_tumor).sum()
    union_whole = pred_whole_tumor.sum() + label_whole_tumor.sum()

    whole_tumor_dice = 2.0 * intersection_whole / union_whole
    dice_scores[:, num_classes - 1] = whole_tumor_dice

    # 返回单类别和 Whole Tumor 的 Dice 分数
    return dice_scores.cpu().numpy()

def iou_score(preds, labels, num_classes):
    batch_size = preds.shape[0]
    iou_scores = torch.zeros(batch_size, num_classes, device=preds.device)

    # 单类别的 IoU 计算
    for cls in range(1, num_classes):  # 排除背景类
        pred_cls = (preds == cls).float()
        label_cls = (labels == cls).float()

        intersection = (pred_cls * label_cls).sum()  # 批量计算交集
        union = pred_cls.sum() + label_cls.sum() - intersection  # 批量计算并集

        iou = intersection / union
        iou_scores[:, cls - 1] = iou.nan_to_num(nan=float('nan'))  # 处理 nan 值

    # Whole Tumor IoU 计算 (由标签 1, 2, 3 组成)
    pred_whole_tumor = ((preds == 1) | (preds == 2) | (preds == 3)).float()
    label_whole_tumor = ((labels == 1) | (labels == 2) | (labels == 3)).float()

    intersection_whole = (pred_whole_tumor * label_whole_tumor).sum()
    union_whole = pred_whole_tumor.sum() + label_whole_tumor.sum() - intersection_whole

    whole_tumor_iou = intersection_whole / union_whole
    iou_scores[:, num_classes - 1] = whole_tumor_iou.nan_to_num(nan=float('nan'))  # 处理 nan 值

    # 返回单类别和 Whole Tumor 的 IoU 分数
    return iou_scores.cpu().numpy()

def hd95(preds, labels, spacing=(1, 1, 1), max_classes=5):
    batch_size = preds.shape[0]
    hd95_scores = torch.full((batch_size, max_classes), float('nan'), device=preds.device)

    for cls in range(1, max_classes):  # 排除背景类
        pred_cls = (preds == cls).cpu().numpy()  # 转为 NumPy 布尔张量
        label_cls = (labels == cls).cpu().numpy()

        for b in range(batch_size):
            pred_mask = pred_cls[b]
            label_mask = label_cls[b]

            if not np.any(pred_mask) and not np.any(label_mask):
                hd95_value = 0.0
            elif not np.any(pred_mask) or not np.any(label_mask):
                hd95_value = float('nan')
            else:
                hd95_value = __compute_hd95(pred_mask, label_mask, spacing)
            hd95_scores[b, cls - 1] = hd95_value  # 保存结果

    # Whole Tumor HD95
    pred_whole_tumor = ((preds == 1) | (preds == 2) | (preds == 3)).cpu().numpy()
    label_whole_tumor = ((labels == 1) | (labels == 2) | (labels == 3)).cpu().numpy()

    whole_tumor_hd95 = []
    for b in range(batch_size):
        pred_mask = pred_whole_tumor[b]
        label_mask = label_whole_tumor[b]

        if not np.any(pred_mask) and not np.any(label_mask):
            hd95_value = 0.0
        elif not np.any(pred_mask) or not np.any(label_mask):
            hd95_value = float('nan')
        else:
            hd95_value = __compute_hd95(pred_mask, label_mask, spacing)
        hd95_scores[b, max_classes - 1] = hd95_value

    return hd95_scores.cpu().numpy()

def __compute_hd95(pred, gt, spacing):
    # Compute edges
    pred_edge = pred ^ binary_erosion(pred)
    gt_edge = gt ^ binary_erosion(gt)
    if np.sum(pred_edge) == 0:
        pred_edge = pred
    if np.sum(gt_edge) == 0:
        gt_edge = gt
    # Distance transform
    dt_pred = distance_transform_edt(~pred_edge, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_edge, sampling=spacing)
    # Surface distances
    sds_pred = dt_gt[pred_edge]
    sds_gt = dt_pred[gt_edge]
    sds = np.concatenate([sds_pred, sds_gt])
    if len(sds) == 0:
        return 0.0
    else:
        hd95 = np.percentile(sds, 95)
        return hd95
    

def calculate_1_96SE(data):
    """
    计算一个形状为 (135, 5) 的数组每列的 1.96SE，忽略 NaN 值。
    
    参数:
    data (numpy.ndarray): 输入数组，形状为 (135, 5)。
    
    返回:
    numpy.ndarray: 一个形状为 (1, 5) 的数组，每列的 1.96SE。
    """
    # 样本数（每列非 NaN 值的数量）
    n = np.sum(~np.isnan(data), axis=0)
    
    # 每列的标准差，忽略 NaN
    std_dev = np.nanstd(data, axis=0, ddof=1)
    
    # 每列的标准误差
    se = std_dev / np.sqrt(n)
    
    # 每列的 1.96SE
    ci_95 = 1.96 * se
    
    # 返回形状为 (1, 5) 的数组
    return ci_95[np.newaxis, :]


def main():
    device = "cuda:0"
    #model = unet.UKAN(5)
    roi = (192, 160, 160)
    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=5,
        feature_size=48,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        use_checkpoint=True,
    )
    
    model_path = "./results/swinUNet.pth"
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    model.cuda()
    patch_size = (192, 160, 160)
    dice_all = []
    iou_all = []
    hd95_all = []
    num_classes = 5

    val_dataset = BraTS(args.data_path, args.valid_txt, transform=transforms.Compose([
        RandomCrop(patch_size),
        ToTensor()
    ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)
    output_list = []
    name_list = []
    for batch in val_loader:
        input, labels, name = batch
        #print(name)
        name_list.append(name)
        input = input.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            batch_output = model(input)
            probs = F.softmax(batch_output, dim=1)
            preds = torch.argmax(probs, dim=1)  # Shape: [batch_size, 192, 160, 160]
            #print(preds.shape)
            #print(labels.shape)

            # Compute metrics
            dice = dice_coefficient(preds, labels, num_classes)
            #print("dice:", dice)
            iou = iou_score(preds, labels, num_classes)
            #print("iou:", iou)
            hd95_values = hd95(preds, labels)
            #print("hd95:", hd95_values)

            dice_all.append(dice)    # Each element is a list of per-class metrics
            iou_all.append(iou)
            hd95_all.append(hd95_values)
            #print("out:", batch_output.shape)
            batch_output = batch_output.permute(0, 1, 3, 2, 4)
            output_list.append(batch_output.cpu().numpy())

    # Concatenate along the sample axis
    #for idx, array in enumerate(hd95_all):
        #print(f"Array {idx} shape: {array.shape}")
    dice_all = np.concatenate(dice_all, axis=0)    # Shape: [total_samples, num_classes - 1]
    iou_all = np.concatenate(iou_all, axis=0)
    hd95_all = np.concatenate(hd95_all, axis=0)

    # Compute mean metrics per class
    dice_mean = np.nanmean(dice_all, axis=0)  # Mean dice per class
    iou_mean = np.nanmean(iou_all, axis=0)
    hd95_mean = np.nanmean(hd95_all, axis=0)

    # Compute overall mean (across classes)
    dice_mean_overall = np.nanmean(dice_mean)
    iou_mean_overall = np.nanmean(iou_mean)
    hd95_mean_overall = np.nanmean(hd95_mean)

    # Store results in variables
    dice_results = {'per_class': dice_mean, 'overall': dice_mean_overall}
    print("dice_results:", dice_results)
    print("Dice SE:")
    print(calculate_1_96SE(dice_all))
    iou_results = {'per_class': iou_mean, 'overall': iou_mean_overall}
    print("iou_results:", iou_results)
    print("iou SE:")
    print(calculate_1_96SE(iou_all))
    hd95_results = {'per_class': hd95_mean, 'overall': hd95_mean_overall}
    print("hd95_results:", hd95_results)
    print("hd95 SE:")
    print(calculate_1_96SE(hd95_all))

    '''xx
    # Print results
    for i in range(num_classes - 1):  # Exclude background class 0
        print(f"Class {i+1} Dice: {dice_mean[i]:.4f}, IoU: {iou_mean[i]:.4f}, HD95: {hd95_mean[i]:.4f}")

    print(f"Overall Dice: {dice_mean_overall:.4f}, Overall IoU: {iou_mean_overall:.4f}, Overall HD95: {hd95_mean_overall:.4f}")
    '''

    target_shape = (182, 218, 182)

    def process_output(output):
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)
        # Get the predicted class for each voxel
        labels = torch.argmax(probs, dim=1)
        # Squeeze to remove the channel dimension (1, depth, height, width)
        labels = labels.squeeze(0).cpu().numpy()  # Shape: (160, 192, 160)
        return labels

    def clean_filenames(namelist):
        cleaned_names = [name[0].strip("(),'").split('/')[-1] for name in namelist]
        return cleaned_names

    def resize_and_pad(labels, target_shape):
        current_shape = labels.shape

        # Calculate padding and cropping for center padding
        pad_widths = [(max(0, (target - current) // 2), max(0, (target - current + 1) // 2))
                      for current, target in zip(current_shape, target_shape)]
        crop_widths = [(max(0, (current - target) // 2), max(0, (current - target + 1) // 2))
                       for current, target in zip(current_shape, target_shape)]

        # Pad and crop
        labels = np.pad(labels, pad_widths, mode='constant', constant_values=0)
        slices = [slice(crop[0], crop[0] + target) for crop, target in zip(crop_widths, target_shape)]
        resized = labels[tuple(slices)]

        return resized

    def save_as_nii(data, filename, origin):
        #print("STEP2")
        affine = np.eye(4)
        affine[:3, 3] = origin
        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, filename)


    processed_outputs = []
    for output in output_list:
        output = torch.tensor(output)
        labels = process_output(output)
        #print(labels.shape)
        resized_labels = resize_and_pad(labels, target_shape)
        processed_outputs.append(resized_labels)

    cleaned_names = clean_filenames(name_list)
    origin = [-90, 126, -72]
    for i, processed_output in enumerate(processed_outputs):
        filename = f"./val_output/{cleaned_names[i]}.nii.gz"
        #print("STEP0")
        processed_output = processed_output.astype(np.float32)
        #"STEP1")
        save_as_nii(processed_output, filename, origin)

    def zip_folder(folder_path, output_path):
        """
        Compress all files in the specified folder into a single .zip file.

        Parameters:
        - folder_path (str): The path of the folder to compress.
        - output_path (str): The path of the output .zip file.
        """
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)

    # Example usage

    print("Processing and saving complete.")


if __name__ == '__main__':
    main()
