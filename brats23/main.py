import numpy as np # linear algebra

import os
import h5py
import ukantrans
import ukan
from aunet import AttentionUNet
from tqdm import tqdm
from scipy.ndimage import rotate
import gc
from torch.profiler import profile, record_function, ProfilerActivity
import random
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from swin_unetr import SwinUNETR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from einops import rearrange
import torch
print(torch.__version__)
print(torch.version.cuda)


class Config:
    def __init__(self):
        self.num_classes = 4
        self.seed = 21
        self.epochs = 100
        self.warmup_epochs = 10
        self.batch_size = 1
        self.lr = 0.001
        self.min_lr = 0.0005
        self.data_path = '/home/tianzetang/brats/data23/dataset'
        #self.data_path = '/scratch/tt2631/brats24/data23/dataset'
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
        #print(d, h, w)
        new_d, new_h, new_w = self.output_size
        
        if d - new_d < 0 or h - new_h < 0 or w - new_w < 0:
            raise ValueError("Output size must be smaller than the dimensions of the input image")

        d1 = 24
        h1 = 24
        w1 = 13

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
            self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]
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
    label1: NCR
    label2: ED
    label3: ET
    dice1(ET):label 3
    dice2(TC):label 1+3
    dice3(WT):label 1+2+3
    '''
    output = torch.argmax(output, dim=1)
    #print(output.shape)
    # dice1 = Dice((output == 3).float(), (target == 3).float())
    # dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    # dice3 = Dice((output != 0).float(), (target != 0).float())

    diceNCR = Dice((output == 1).float(), (target == 1).float())
    diceED = Dice((output == 2).float(), (target == 2).float())
    diceET = Dice((output == 3).float(), (target == 3).float())
    # Combined regions
    diceTC = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())  # ET + NETC
    diceWT = Dice(((output == 1) | (output == 2) | (output == 3)).float(), ((target == 1) | (target == 2) | (target == 3)).float())  # ET + SNFH + NETC

    return diceNCR, diceED, diceET, diceTC, diceWT


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 1

        input1 = F.softmax(input, dim=1)
        #input1 = F.sigmoid(input)
        target1 = F.one_hot(target, self.n_classes)
        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = (2.0 * inter + smooth) / union

        celoss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * celoss + (1 - dice) * self.alpha

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
    diceNCR_train = []
    diceED_train = []
    diceET_train = []
    diceTC_train = []
    diceWT_train = []
    pbar = tqdm(train_loader)
    for it,(images,masks) in enumerate(pbar):
    #for it,(images,masks) in enumerate(train_loader):
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

        dice1, dice2, dice3, dice4, dice5 = cal_dice(outputs, masks)
        print("train_loss:", loss.item())

        #pbar.desc = "loss: {:.3f} ".format(loss.item())

        running_loss.append(loss.item())
        diceNCR_train.append(dice1.item())
        diceED_train.append(dice2.item())
        diceET_train.append(dice3.item())
        diceTC_train.append(dice4.item())
        diceWT_train.append(dice5.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(running_loss)/len(running_loss)
    avg_dice1 = sum(diceNCR_train) / len(diceNCR_train)
    avg_dice2 = sum(diceED_train) / len(diceED_train)
    avg_dice3 = sum(diceET_train) / len(diceET_train)
    avg_dice4 = sum(diceTC_train) / len(diceTC_train)
    avg_dice5 = sum(diceWT_train) / len(diceWT_train)

    return {'loss': avg_loss, 'diceNCR': avg_dice1, 'diceED': avg_dice2, 'diceET': avg_dice3,
            'diceTC': avg_dice4, 'diceWT': avg_dice5}


def val_loop(model,criterion,val_loader,device):
    model.eval()
    running_loss = []
    diceNCR_val = []
    diceED_val = []
    diceET_val = []
    diceTC_val = []
    diceWT_val = []
    #pbar = tqdm(val_loader)
    with torch.no_grad():
        #for images, masks in pbar:
        for images, masks in (val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            print("val_loss:", loss.item())
            dice1, dice2, dice3, dice4, dice5= cal_dice(outputs, masks)

            running_loss.append(loss.item())
            diceNCR_val.append(dice1.item())
            diceED_val.append(dice2.item())
            diceET_val.append(dice3.item())
            diceTC_val.append(dice4.item())
            diceWT_val.append(dice5.item())
            #pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    avg_loss = sum(running_loss) / len(running_loss)
    avg_dice1 = sum(diceNCR_val) / len(diceNCR_val)
    avg_dice2 = sum(diceED_val) / len(diceED_val)
    avg_dice3 = sum(diceET_val) / len(diceET_val)
    avg_dice4 = sum(diceTC_val) / len(diceTC_val)
    avg_dice5 = sum(diceWT_val) / len(diceWT_val)
    return {'loss': avg_loss, 'diceNCR': avg_dice1, 'diceED': avg_dice2, 'diceET': avg_dice3,
            'diceTC': avg_dice4, 'diceWT': avg_dice5}



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
        
        info2 = "Train--Dice: NCR: {:.3f} ED: {:.3f} ET: {:.3f} TC:{:.3f} WT:{:.3f} ".format(train_metrics['diceNCR'],train_metrics['diceED'],train_metrics['diceET'],
                                                        train_metrics['diceTC'],train_metrics['diceWT'])
        
        info3 = "Valid--Dice: NCR: {:.3f} ED: {:.3f} ET: {:.3f} TC:{:.3f} WT:{:.3f} ".format(val_metrics['diceNCR'],val_metrics['diceED'],val_metrics['diceET'],
                                                        val_metrics['diceTC'],val_metrics['diceWT'])

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
    patch_size = (192, 192, 128)
    train_dataset = BraTS(args.data_path,args.train_txt,transform=transforms.Compose([
        RandomCrop(patch_size),
        RandomFlip(),
        AddNoise(),
        RandomRotate(),
        RandomContrast(),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path,args.valid_txt,transform=transforms.Compose([
        RandomCrop(patch_size),
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=16,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False,
                            pin_memory=True)
    #test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             #pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    model = ukantrans.UKAN(4)
    '''
    model = SwinUNETR(
            in_channels=4,
            out_channels=4,
            img_size=[192, 192, 128],
            drop_rate = 0.1,
            attn_drop_rate = 0.1,
            dropout_path_rate = 0.1,
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

    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.2, 0.2, 0.2])).to(device)
    #criterion = Loss1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.005)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = cosine_scheduler(base_value=args.lr,final_value=args.min_lr,epochs=args.epochs,
                                 niter_per_ep=len(train_loader),warmup_epochs=args.warmup_epochs,start_warmup_value=0.0005)

    # load training model
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train_metrics_all, val_metrics_all = train(model,optimizer,scheduler,criterion,train_loader,val_loader,args.epochs,device,train_log=args.train_log)

    #metrics1 = val_loop(model, criterion, train_loader, device)
    #metrics2 = val_loop(model, criterion, val_loader, device)
    #metrics3 = val_loop(model, criterion, test_loader, device)

    # Finally, evaluate all the data again. 
    # Note that the model parameters used here are from the end of the training
    # print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    #print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3'],
                                                                          #metrics2['dice4'], metrics2['dice5'], metrics2['dice6']))
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
    train_et = [entry['diceNCR'] for entry in train_metrics_all]
    val_et = [entry['diceNCR'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training NCR')
    plt.plot(val_et, label='Validation NCR')

    plt.xlabel('Epoch')
    plt.ylabel('NCR Accracy')
    plt.title('Training and Validation NCR Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "diceNCR.png"))

    ##dice2
    train_et = [entry['diceED'] for entry in train_metrics_all]
    val_et = [entry['diceED'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training ED')
    plt.plot(val_et, label='Validation ED')

    plt.xlabel('Epoch')
    plt.ylabel('ED Accracy')
    plt.title('Training and Validation ED Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "diceED.png"))

    ##dice3 SNFH
    train_et = [entry['diceET'] for entry in train_metrics_all]
    val_et = [entry['diceET'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training ET')
    plt.plot(val_et, label='Validation ET')

    plt.xlabel('Epoch')
    plt.ylabel('ET Accracy')
    plt.title('Taining and Validation ET Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "diceET.png"))

    ##dice4 RC
    train_et = [entry['diceTC'] for entry in train_metrics_all]
    val_et = [entry['diceTC'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training TC')
    plt.plot(val_et, label='Validation TC')

    plt.xlabel('Epoch')
    plt.ylabel('TC Accracy')
    plt.title('Taining and Validation TC Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "diceTC.png"))

    ##dice5 (ET+NETC)
    train_et = [entry['diceWT'] for entry in train_metrics_all]
    val_et = [entry['diceWT'] for entry in val_metrics_all]

    plt.figure(figsize=(10, 5))
    plt.plot(train_et, label='Training WT')
    plt.plot(val_et, label='Validation WT')

    plt.xlabel('Epoch')
    plt.ylabel('WT Accracy')
    plt.title('Taining and Validation WT Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_path, "diceWT.png"))

    ## dice mean
    train_mean = [(entry['diceWT']+entry['diceTC']+entry['diceET']+entry['diceED']+entry['diceNCR'])/5 for entry in train_metrics_all]
    val_mean = [(entry['diceWT']+entry['diceTC']+entry['diceET']+entry['diceED']+entry['diceNCR'])/5 for entry in val_metrics_all]

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