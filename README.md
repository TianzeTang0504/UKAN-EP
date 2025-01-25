# Brats 2024 Task 1 Using UKAN and Other Models
Code for Brats2024 Task 1: Segmentation - Adult Glioma Post Treatment
This repository contains a complete pipeline from data pre-processing to model validation using Brats 2024 Task 1 data.

## Data overview
Brats 2024 Task 1 contains 1350 brain MRI nii files. All BraTS mpMRI scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple data contributing institutions. The ground truth data was created after preprocessing, including co-registration to the same anatomical template, interpolation to the same resolution (1 mm3), and skull stripping.

All the imaging datasets have been annotated manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuroradiologists. Annotations comprise the enhancing tissue (ET — label 3), the surrounding non-enhancing FLAIR hyperintensity (SNFH) — label 2), the non-enhancing tumor core (NETC — label 1), and the resection cavity (RC - label 4) as described in the latest BraTS summarizing paper, except that the resection cavity has been incorporated subsequent to the paper's release.

Image example:

![](https://github.com/TianzeTang0504/brats24/blob/main/pngs/datanii.png)

All images are organized into separate folders like this:

.

├── BraTS-GLI-00000-000

│   ├── BraTS-GLI-00000-000-seg.nii.gz

│   ├── BraTS-GLI-00000-000-t1c.nii.gz

│   ├── BraTS-GLI-00000-000-t1n.nii.gz

│   ├── BraTS-GLI-00000-000-t2f.nii.gz

│   └── BraTS-GLI-00000-000-t2w.nii.gz

├── BraTS-GLI-00001-000

│   ├── BraTS-GLI-00001-000-seg.nii.gz

│   ├── BraTS-GLI-00001-000-t1c.nii.gz

│   ├── BraTS-GLI-00001-000-t1n.nii.gz

│   ├── BraTS-GLI-00001-000-t2f.nii.gz

│   └── BraTS-GLI-00001-000-t2w.nii.gz

├── BraTS-GLI-00002-000

...

You can download all images from [Brats 2024 website](https://www.synapse.org/Synapse:syn53708249/wiki/627500) which is almost 70GB.

## Data pre-processing

We firstly combine the 4 kinds of MRI images into a single 4 channels image which size is 4\*218\*182\*182 and then normalize none-background area into 0-1. We make these combined images into h5 file and using txt file to record names of these h5 files. This is a really efficient way to organize dataset. You can find these parts in ./pre_processing. This idea and code are from [icerain-alt](https://github.com/icerain-alt/brats-unet).

Before training, we do some regular data augmentation like adding noise, random flip and so on.

## Metrics

### Loss Function

We use Soft Dice and CE lose together, α is dynamic:

![](https://github.com/TianzeTang0504/Brats-2024-Task1/blob/main/pngs/loss.png)

## Models

### U-KAN ＆ U-KAN-EP

The UKAN model we use is from [CUHK-AIM-Group](https://github.com/CUHK-AIM-Group/U-KAN) work. Their original model is only for 2D images. We change all the blocks into 3D such as Conv2d-Conv3d. UKAN-EP is based on that.

![](https://github.com/TianzeTang0504/Brats-2024-Task1/blob/main/pngs/ukan_structure.png)

### UNET ＆ Attention-UNET

The model we use is also from [icerain-alt](https://github.com/icerain-alt/brats-unet) work.

### Swin-UNETR

Swin-UNETR is from [MONAI](https://github.com/Project-MONAI/MONAI/tree/dev).

## Results

KAN based model have really good efficiency and relatively good scores.

![](https://github.com/TianzeTang0504/Brats_2024-3D_Brain_MRI_Segmentation-UKAN/blob/main/pngs/time.png)

![](https://github.com/TianzeTang0504/Brats_2024-3D_Brain_MRI_Segmentation-UKAN/blob/main/pngs/dice.png)

![](https://github.com/TianzeTang0504/Brats_2024-3D_Brain_MRI_Segmentation-UKAN/blob/main/pngs/hd95.png)
