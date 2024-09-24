import argparse
import os
from collections import OrderedDict
import glob
import random
import numpy as np
import json
import torch.nn.functional as F
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from swin_unetr import SwinUNETR
import h5py


import numpy as np
import os
import zipfile
import nibabel as nib
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from swin_unetr import SwinUNETR
import ukan
import ukanSE
from unet import UNet
from aunet import AttentionUNet
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
        self.data_path = 'D:\\brats2024\\data\\valdata'  # train_set['out']
        # self.data_path = '/scratch/tt2631/brats24/data/dataset'  # train_set['out']
        self.train_txt = '.\\train_split.txt'
        self.valid_txt = '.\\val.txt'
        # self.test_txt = '.\\test_split.txt'
        self.train_log = '.\\output\\UNet.txt'
        self.weights = '.\\output\\UNet.pth'
        self.save_path = '.\\output'


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
        image = sample['image']

        _, d, h, w = image.shape
        new_d, new_h, new_w = self.output_size

        if d - new_d < 0 or h - new_h < 0 or w - new_w < 0:
            raise ValueError("Output size must be smaller than the dimensions of the input image")

        d1 = 13
        h1 = 11
        w1 = 11

        image = image[:, d1:d1 + new_d, h1:h1 + new_h, w1:w1 + new_w]

        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        image = torch.from_numpy(image).float()

        return {'image': image}


class BraTS(Dataset):
    def __init__(self, data_path, file_path, transform=None):
        with open(file_path, 'r') as f:
            self.paths = [os.path.join(data_path, x.strip() + '-mri_norm2.h5') for x in f.readlines()]
        with open(file_path, 'r') as f:
            self.paths_2 = [os.path.join(data_path, x.strip()) for x in f.readlines()]
        self.transform = transform

    def __getitem__(self, item):
        #print(item)
        #print(len(self.paths))
        #print(len(self.paths_2))
        h5f = h5py.File(self.paths[item], 'r')
        image = h5f['image'][:]
        sample_name = self.paths_2[item]
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample_name

    def __len__(self):
        return len(self.paths)

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



def main():
    device = "cuda:0"
    model = ukan.UKAN(5)
    '''
    model = ukanSE(5)
    model = UNet(4, 5)
    model = AttentionUNet(4, 5)
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
    '''
    model_path = "D:\\brats2024\\paper\\swin\\swin.pth"
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    model.cuda()
    patch_size = (192, 160, 160)
    val_dataset = BraTS(args.data_path, args.valid_txt, transform=transforms.Compose([
        RandomCrop(patch_size),
        ToTensor()
    ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=4)
    output_list = []
    name_list = []
    for batch in val_loader:
        input, name = batch
        print(name)
        name_list.append(name)
        input = input.to(device)
        with torch.no_grad():
            batch_output = model(input)
            print("out:", batch_output.shape)
            batch_output = batch_output.permute(0, 1, 3, 2, 4)
            output_list.append(batch_output.cpu().numpy())

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
        cleaned_names = [name[0].strip("(),'").split('\\')[-1] for name in namelist]
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
        print("STEP2")
        affine = np.eye(4)
        affine[:3, 3] = origin
        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, filename)


    processed_outputs = []
    for output in output_list:
        output = torch.tensor(output)
        labels = process_output(output)
        print(labels.shape)
        resized_labels = resize_and_pad(labels, target_shape)
        processed_outputs.append(resized_labels)

    cleaned_names = clean_filenames(name_list)
    origin = [-90, 126, -72]
    for i, processed_output in enumerate(processed_outputs):
        filename = f".\\val_output\\{cleaned_names[i]}.nii.gz"
        print("STEP0")
        processed_output = processed_output.astype(np.float32)
        print("STEP1")
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
    folder_to_zip = 'D:\\brats2024\\unet\\val_output'
    output_zip_file = '.\\VAL.zip'
    zip_folder(folder_to_zip, output_zip_file)

    print("Processing and saving complete.")


if __name__ == '__main__':
    main()
