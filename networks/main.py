import numpy as np # linear algebra

import os
import h5py
import ukan
from swin_unetr import SwinUNETR
import ukanSE
from unet import UNet
from aunet import AttentionUNet
from scipy.ndimage import rotate
import gc
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from einops import rearrange

class Config:
    def __init__(self):
        self.num_classes = 5
        self.seed = 21
        self.epochs = 200
        self.warmup_epochs = 10
        self.batch_size = 1
        self.lr = 0.005
        self.min_lr = 0.001
        #self.data_path = 'D:\\brats2024\\data\\dataset'#train_set['out']
        self.data_path = '/scratch/tt2631/brats24/data/dataset'
        self.train_txt = './train_split.txt'
        self.valid_txt = './valid_split.txt'
        #self.test_txt = './test_split.txt'
        self.train_log = './output/UNet.txt'
        self.weights = './output/UNet.pth'
        self.save_path = './output'

args = Config()

random.seed(21)
np.random.seed(21)

if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for training.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training will be on CPU.")
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


class RandomFlip:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.rand() > 0.5:
            # Flip along the x-axis
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        if np.random.rand() > 0.5:
            # Flip along the y-axis
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if np.random.rand() > 0.5:
            # Flip along the z-axis
            image = np.flip(image, axis=3).copy()
            label = np.flip(label, axis=2).copy()
        return {'image': image, 'label': label}


class AddNoise:
    def __init__(self, noise_variance=0.01):
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.random.normal(0, self.noise_variance, image.shape)
        image += noise
        return {'image': image, 'label': label}


class RandomRotate:
    def __init__(self, angle_range=(-10, 10)):
        self.angle_range = angle_range

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = np.random.uniform(*self.angle_range)

        # Rotate image
        image_axes = random.choice([(1, 2), (1, 3), (2, 3)])
        image = rotate(image, angle, axes=image_axes, reshape=False, order=1, mode='nearest')

        # Rotate label
        label_axes = (image_axes[0] - 1, image_axes[1] - 1)  # Adjust axes for 3D label
        label = rotate(label, angle, axes=label_axes, reshape=False, order=0, mode='nearest')

        return {'image': image, 'label': label}


class RandomContrast:
    def __init__(self, contrast_range=(0.8, 1.2)):
        self.contrast_range = contrast_range

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        contrast_factor = np.random.uniform(*self.contrast_range)
        mean = np.mean(image, axis=(1, 2, 3), keepdims=True)
        image = (image - mean) * contrast_factor + mean
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
            self.paths = [os.path.join(data_path, x.strip() + '-mri_norm2.h5') for x in f.readlines()]
        self.transform = transform

    def __getitem__(self, item):
        h5f = h5py.File(self.paths[item], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


def Dice(output, target, eps=1e-4):
    inter = torch.sum(output * target, dim=(1, 2, -1)) + eps
    union = torch.sum(output, dim=(1, 2, -1)) + torch.sum(target, dim=(1, 2, -1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    label1: NETC 
    label2: SNFH
    label3: ET
    label4: RC
    dice1(ET):label3
    dice2(NETC):label1
    dice3(SNFH):label2
    dice4(RC): label4
    dice5(ET+NETC):label1+label3
    dice6(ET+SNFH+NETC):label1+label2+label3
    '''
    output = torch.argmax(output, dim=1)
    #print(output.shape)
    # dice1 = Dice((output == 3).float(), (target == 3).float())
    # dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    # dice3 = Dice((output != 0).float(), (target != 0).float())

    dice1 = Dice((output == 3).float(), (target == 3).float())  # ET
    dice2 = Dice((output == 1).float(), (target == 1).float())  # NETC
    dice3 = Dice((output == 2).float(), (target == 2).float())  # SNFH
    dice4 = Dice((output == 4).float(), (target == 4).float())  # RC
    # Combined regions
    dice5 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())  # ET + NETC
    dice6 = Dice(((output == 1) | (output == 2) | (output == 3)).float(), ((target == 1) | (target == 2) | (target == 3)).float())  # ET + SNFH + NETC

    return dice1, dice2, dice3, dice4, dice5, dice6


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target,self.n_classes)
        input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


# The learning rate is updated at each iteration, not just at each epoch.
# Access the learning rate schedule at the current iteration index (scheduler[iter]).
# Update the learning rate for the optimizer's parameter group.
# help the model converge more smoothly and potentially achieve better performance.
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def train_loop(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    running_loss = []
    dice1_train = []
    dice2_train = []
    dice3_train = []
    dice4_train = []
    dice5_train = []
    dice6_train = []
    #pbar = tqdm(train_loader)
    #for it,(images,masks) in enumerate(pbar):
    for it,(images,masks) in enumerate(train_loader):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]
        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device),masks.to(device)
        #print("images_shape:", images.shape)
        #print("masks_shape:", masks.shape)
        # [b,4,128,128,128], 4 segmentations
        outputs = model(images)
        #print("outputs_shape:", outputs.shape)
        # outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, masks)

        dice1, dice2, dice3, dice4, dice5, dice6 = cal_dice(outputs, masks)
        print("train_loss:", loss.item())

        #pbar.desc = "loss: {:.3f} ".format(loss.item())

        running_loss.append(loss.item())
        dice1_train.append(dice1.item())
        dice2_train.append(dice2.item())
        dice3_train.append(dice3.item())
        dice4_train.append(dice4.item())
        dice5_train.append(dice5.item())
        dice6_train.append(dice6.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(running_loss)/len(running_loss)
    avg_dice1 = sum(dice1_train) / len(dice1_train)
    avg_dice2 = sum(dice2_train) / len(dice2_train)
    avg_dice3 = sum(dice3_train) / len(dice3_train)
    avg_dice4 = sum(dice4_train) / len(dice4_train)
    avg_dice5 = sum(dice5_train) / len(dice5_train)
    avg_dice6 = sum(dice6_train) / len(dice6_train)

    return {'loss': avg_loss, 'dice1': avg_dice1, 'dice2': avg_dice2, 'dice3': avg_dice3,
            'dice4': avg_dice4, 'dice5': avg_dice5, 'dice6': avg_dice6}


def val_loop(model,criterion,val_loader,device):
    model.eval()
    running_loss = []
    dice1_val = []
    dice2_val = []
    dice3_val = []
    dice4_val = []
    dice5_val = []
    dice6_val = []
    #pbar = tqdm(val_loader)
    with torch.no_grad():
        #for images, masks in pbar:
        for images, masks in (val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            print("val_loss:", loss.item())
            dice1, dice2, dice3, dice4, dice5, dice6 = cal_dice(outputs, masks)

            running_loss.append(loss.item())
            dice1_val.append(dice1.item())
            dice2_val.append(dice2.item())
            dice3_val.append(dice3.item())
            dice4_val.append(dice4.item())
            dice5_val.append(dice5.item())
            dice6_val.append(dice6.item())
            #pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    avg_loss = sum(running_loss) / len(running_loss)
    avg_dice1 = sum(dice1_val) / len(dice1_val)
    avg_dice2 = sum(dice2_val) / len(dice2_val)
    avg_dice3 = sum(dice3_val) / len(dice3_val)
    avg_dice4 = sum(dice4_val) / len(dice4_val)
    avg_dice5 = sum(dice5_val) / len(dice5_val)
    avg_dice6 = sum(dice6_val) / len(dice6_val)
    return {'loss': avg_loss, 'dice1': avg_dice1, 'dice2': avg_dice2, 'dice3': avg_dice3,
            'dice4': avg_dice4, 'dice5': avg_dice5, 'dice6': avg_dice6}



def train(model,optimizer,scheduler,criterion,train_loader,
          val_loader,epochs,device,train_log,valid_loss_min=999.0):
    train_metrics_all = []
    val_metrics_all = []
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model,optimizer,scheduler,criterion,train_loader,device,e)
        train_metrics_all.append(train_metrics)
        # eval for epoch
        val_metrics = val_loop(model,criterion,val_loader,device)
        val_metrics_all.append(val_metrics)
        info1 = "Epoch:[{}/{}] valid_loss_min: {:.3f} train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,valid_loss_min,train_metrics["loss"],val_metrics["loss"])
        
        info2 = "Train--Dice: ET: {:.3f} NETC: {:.3f} SNFH: {:.3f} RC:{:.3f} ET+NETC:{:.3f} ET+SNFH+NETC:{:.3f} ".format(train_metrics['dice1'],train_metrics['dice2'],train_metrics['dice3'],
                                                           train_metrics['dice4'],train_metrics['dice5'],train_metrics['dice6'])
        
        info3 = "Valid--Dice: ET: {:.3f} NETC: {:.3f} SNFH: {:.3f} RC:{:.3f} ET+NETC:{:.3f} ET+SNFH+NETC:{:.3f} ".format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'],
                                                           val_metrics['dice4'],val_metrics['dice5'],val_metrics['dice6'])

        print(info1)
        print(info2)
        print(info3)
        with open(train_log,'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, args.weights)
        else:
            torch.save(save_file,os.path.join(args.save_path,'checkpoint{}.pth'.format(e+1)))
    print("Finished Training!")
    return train_metrics_all, val_metrics_all


def main_train(args):
    #torch.manual_seed(args.seed)  # Set the seed for the CPU to ensure reproducible results
    #torch.cuda.manual_seed_all(args.seed)  # Set the seed for all GPUs to ensure reproducible results

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get datasets
    patch_size = (192, 160, 160)
    train_dataset = BraTS(args.data_path,args.train_txt,transform=transforms.Compose([
        #RandomRotFlip(), 
        RandomCrop(patch_size),
        RandomFlip(),
        AddNoise(),
        RandomRotate(),
        RandomContrast(),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path,args.valid_txt,transform=transforms.Compose([
        RandomCrop(patch_size),
        #CenterCrop(patch_size),
        ToTensor()
    ]))
    #test_dataset = BraTS(args.data_path,args.test_txt,transform=transforms.Compose([
        #RandomCrop(patch_size),
        #CenterCrop(patch_size),
        #ToTensor()
    #]))
    # a glance at dataset
    d1 = train_dataset[0]
    image, label = d1
    print(image.shape)
    print(label.shape)
    #print(np.unique(label))

    # data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                            pin_memory=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             #pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    # img,label = train_dataset[0]
    # # label1: NETC 
    # # label2: SNFH
    # # label3: ET
    # # label4: RC
    # # dice1(ET):label3
    # # dice2(NETC):label1
    # # dice3(SNFH):label2
    # # dice4(RC): label4
    # # dice5(ET+NETC):label1+label3
    # # dice6(ET+SNFH+NETC):label1+label2+label3
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
    #model_path = "./UNet_39.pth"
    #checkpoint = torch.load(model_path)
    #state_dict = checkpoint['model']
    #model.load_state_dict(state_dict)
    model = model.to(device)
    
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f'Total number of parameters: {num_params / 1e6:.3f} M')

    criterion = Loss(n_classes=5, weight=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = cosine_scheduler(base_value=args.lr,final_value=args.min_lr,epochs=args.epochs,
                                 niter_per_ep=len(train_loader),warmup_epochs=args.warmup_epochs,start_warmup_value=0.005)

    # load training model
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train_metrics_all, val_metrics_all = train(model,optimizer,scheduler,criterion,train_loader,val_loader,args.epochs,device,train_log=args.train_log)

    # metrics1 = val_loop(model, criterion, train_loader, device)
    metrics2 = val_loop(model, criterion, val_loader, device)
    #metrics3 = val_loop(model, criterion, test_loader, device)

    # Finally, evaluate all the data again. 
    # Note that the model parameters used here are from the end of the training
    # print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3'],
                                                                          metrics2['dice4'], metrics2['dice5'], metrics2['dice6']))
    #print("Test  -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3'],
                                                                         #metrics3['dice4'], metrics3['dice5'], metrics3['dice6']))

    return train_metrics_all, val_metrics_all


def main():

    if os.path.isfile('./results/UNet.pth'):
        os.remove('./results/UNet.pth')
    if os.path.isfile('./results/UNet.txt'):
        os.remove('./results/UNet.txt')


    gc.collect()
    torch.cuda.empty_cache()
    train_metrics_all, val_metrics_all = main_train(args)


    ## plots
    save_path = "./output"

    ##loss
    train_losses = [entry['loss'] for entry in train_metrics_all]
    valid_losses = [entry['loss'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss.png"))

    #plt.show()

    ##dice1
    train_et = [entry['dice1'] for entry in train_metrics_all]
    val_et = [entry['dice1'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training ET')
    plt.plot(val_et, label='Validation ET')

    plt.xlabel('Epoch')
    plt.ylabel('ET Accracy')
    plt.title('Training and Validation ET Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice1.png"))

    ##dice2
    train_et = [entry['dice2'] for entry in train_metrics_all]
    val_et = [entry['dice2'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training NETC')
    plt.plot(val_et, label='Validation NETC')

    plt.xlabel('Epoch')
    plt.ylabel('NETC Accracy')
    plt.title('Training and Validation NETC Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice2.png"))

    ##dice3 SNFH
    train_et = [entry['dice3'] for entry in train_metrics_all]
    val_et = [entry['dice3'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training SNFH')
    plt.plot(val_et, label='Validation SNFH')

    plt.xlabel('Epoch')
    plt.ylabel('SNFH Accracy')
    plt.title('Taining and Validation SNFH Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice3.png"))

    ##dice4 RC
    train_et = [entry['dice4'] for entry in train_metrics_all]
    val_et = [entry['dice4'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training RC')
    plt.plot(val_et, label='Validation RC')

    plt.xlabel('Epoch')
    plt.ylabel('RC Accracy')
    plt.title('Taining and Validation RC Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice4.png"))

    ##dice5 (ET+NETC)
    train_et = [entry['dice5'] for entry in train_metrics_all]
    val_et = [entry['dice5'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training ET+NETC')
    plt.plot(val_et, label='Validation ET+NETC')

    plt.xlabel('Epoch')
    plt.ylabel('ET+NETC Accracy')
    plt.title('Taining and Validation ET+NETC Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice5.png"))

    ##dice6 (ET+SNFH+SETC)
    train_et = [entry['dice6'] for entry in train_metrics_all]
    val_et = [entry['dice6'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training ET+SNFH+SETC')
    plt.plot(val_et, label='Validation ET+SNFH+SETC')

    plt.xlabel('Epoch')
    plt.ylabel('ET+SNFH+SETC Accracy')
    plt.title('Taining and Validation ET+SNFH+SETC Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice6.png"))

    ## dice mean
    train_mean = [(entry['dice6']+entry['dice5']+entry['dice4']+entry['dice3']+entry['dice2']+entry['dice1'])/6 for entry in train_metrics_all]
    val_mean = [(entry['dice6']+entry['dice5']+entry['dice4']+entry['dice3']+entry['dice2']+entry['dice1'])/6 for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_mean, label='Training Average')
    plt.plot(val_mean, label='Validation Average')

    plt.xlabel('Epoch')
    plt.ylabel('Average Accracy')
    plt.title('Training and Validation Average Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "dice_mean.png"))

if __name__ == '__main__':
    main()