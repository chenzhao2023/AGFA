from model.AGFANet import AGFANet                           
from skimage import filters
import k3d
import numpy as np
from k3d import matplotlib_color_maps
import torch 
import numpy as np 
from PIL import Image
import glob 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import os
import nibabel as nib
import itk
import pickle
import shutil
import os
import gzip
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import argparse
import torch.distributed as dist
from model.unet3d import UNet3D
# from dataloader.npy_3d_Loader import Data
from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
from torch.optim import lr_scheduler
import datetime
from torch.utils.data import random_split
from utils.sliding_window import sliding_window_3d
import torch.nn as nn
from dataloader.npy_3d_Loader import *
# import pandas as pd
from postprocess.keep_the_largest_area import get_aorta_branch
from postprocess.keep_the_largest_area import backpreprcess as postprocess
from postprocess.get_patch import get_patch_new
from utils.evaluation_metrics3D import metrics_3d, Dice, over_rate, under_rate

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
torch.cuda.empty_cache()

batch_size = 1
batch_size_new=1
num_epochs=1   #  600
patch_size = [128, 160, 160]  #  96, 96, 96
overlap_size=[32, 40, 40]  # 32, 32, 32         # 32, 40, 40
best_score = [0]
args = {
    # 'data_path': 'cta_project/data/npy',
    'epochs': 10,
    'input_shape': (128, 160, 160),
    'snapshot': 10,
    'test_step': 1,
    'model_path': '/home/lxy/lxy/001_CASnet/save_models_randomcrop',
    'batch_size': 1,  # VNet 1 other 2
    'folder': 'folder2',
    'model_name': 'CSNet3D',  #UNet3D   CSNet3D
}
ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])

class CustomDataset(Dataset):
    def __init__(self, data_folder,file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.data_folder=data_folder

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)

        nii_data = itk.array_from_image(itk.imread(file_path))

        if self.transform:
            nii_data = self.transform(nii_data)

        nii_tensor = torch.from_numpy(nii_data).float()
        nii_tensor = nii_tensor.unsqueeze(0)
        # train_mask = nii_tensor.unsqueeze(0)

        return nii_tensor

def save_ckpt(net, iter):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    # torch.save(model.state_dict(),PATH)
    torch.save(net.state_dict(), os.path.join(ckpt_path, date + iter + '.pkl'))
    print("{} Saved model to:{}".format("\u2714", ckpt_path))

class MinMaxScale:
    def __call__(self, img):
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)
        return img
min_max_transform = MinMaxScale()
Test_Model = {'AGFANet': AGFANet,
              'UNet3D': UNet3D
              }

data_folder = '/home/lxy/lxy/data_CCTA/train'
data_folder2 = '/home/lxy/lxy/data_CCTA/trainMask'
file_list_img = [f for f in os.listdir(data_folder) if f.endswith('.nii.gz')]
file_list_mask = [f for f in os.listdir(data_folder2) if f.endswith('.nii.gz')]
file_list_img=sorted(file_list_img)
file_list_mask=sorted(file_list_mask)
print('file_list_img  shape:', len(file_list_img))  # 1000

test_img_list = file_list_img[800:]   
test_mask_list = file_list_mask[800:]

dataset_img = CustomDataset(data_folder,test_img_list, transform=min_max_transform)
dataset_mask = CustomDataset(data_folder2,test_mask_list)

# dataloader
dataloader_img = DataLoader(dataset_img, batch_size=batch_size, shuffle=False)  # here must be False, follow the name
dataloader_mask = DataLoader(dataset_mask, batch_size=batch_size, shuffle=False)
print("len(dataloader_img :",len(dataloader_img))
print("len(dataloader_mask :",len(dataloader_mask))
iter_img = iter(dataloader_img)
iter_mask = iter(dataloader_mask)

#load model
net = Test_Model[args['model_name']](2, 1).to(device)
net = nn.DataParallel(net)

ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])
# modelname = ckpt_path + '/' + 'best_score' + '_checkpoint300epo.pkl'
modelname = ckpt_path + '/' + '2024-04-08-50.pkl'  #  2024-04-08-50

checkpoint = torch.load(modelname)
net.load_state_dict(checkpoint)

DSC_Dice_mean=[]
def Dice(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice
def get_metrics(pred, gt):
    pred[pred > 0] = 255
    gt[gt > 0] = 255
    Ur = under_rate(pred, gt)
    Or = over_rate(pred, gt)
    dice = Dice(pred, gt)
    tp, fn, fp, IoU = metrics_3d(pred, gt)
    return tp, fn, fp, IoU, dice, Or, Ur
def get_prediction(pred):
    pred = torch.argmax(pred, dim=1)
    mask = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    # print(np.max(mask),np.min(mask))
    mask = mask.squeeze(0)  # for CE Loss
    return mask
def MinMaxScale(img):
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)
        return img
# # min_max_transform = MinMaxScale()
# data=MinMaxScale(data)

def merge_patches(patches, volume_size, overlap_size):
    """
    Merge the cropped patches into a complete 3D volume.
    Args:
        patches (np.ndarray): the cropped patches, with shape [num_patches, patch_width, patch_height, patch_depth]
        volume_size (tuple or list): the size of the complete volume, with format [width, height, depth]
        overlap_size (tuple or list): the size of overlap between adjacent patches, with format [overlap_width, overlap_height, overlap_depth]
    Returns:
        np.ndarray: the merged volume, with shape [width, height, depth]
    """

    depth, height, width = volume_size
    patch_depth, patch_height, patch_width = patches.shape[1:]
    overlap_depth, overlap_height, overlap_width = overlap_size
    num_patches_z = (depth - patch_depth) // (patch_depth - overlap_depth) + 1
    num_patches_x = (height - patch_height) // (patch_height - overlap_height) + 1
    num_patches_y = (width - patch_width) // (patch_width - overlap_width) + 1
    
    
    print('merge:', num_patches_z, num_patches_x, num_patches_y)
    merged_volume = np.zeros(volume_size)
    weight_volume = np.zeros(volume_size)   # weight_volume的目的是用于记录每个像素在裁剪过程中被遍历的次数，最后用于求平均值
    idx = 0
    for z in range(num_patches_z):
        for x in range(num_patches_x):
            for y in range(num_patches_y):
                z_start = z * (patch_depth - overlap_depth)
                x_start = x * (patch_height - overlap_height)
                y_start = y * (patch_width - overlap_width)

                merged_volume[z_start:z_start+patch_depth, x_start:x_start+patch_height, y_start:y_start+patch_width] += patches[idx]
                weight_volume[z_start:z_start+patch_depth, x_start:x_start+patch_height, y_start:y_start+patch_width] += 1
                idx += 1
    merged_volume /= (weight_volume + 1e-10)   # 肯定有小数, 有nan 值
    # merged_volume=np.divide(merged_volume,weight_volume)
    # merged_volume=MinMaxScale(merged_volume)
    return merged_volume

def custom_round_array(array):
    result = np.empty_like(array)  
    for i in range(array.shape[0]):  
        for j in range(array.shape[1]):  
            for k in range(array.shape[2]):  
                value = array[i, j, k] 
                integer_part = int(value)  
                decimal_part = value - integer_part  
                if decimal_part < 0.4:
                    result[i, j, k] = integer_part
                else:
                    result[i, j, k] = integer_part + 1
    return result

Dice_mean=[]
Recall_mean=[]
Precision_mean=[]

for batch_idx in range(len(dataloader_img)):  # 10

    test_img = next(iter_img)  # [1, 1, 512, 512, 275]
    test_mask = next(iter_mask)

    print("test_img : ",test_img.shape) # torch.Size([1, 1, 206, 512, 512])
    print("test_mask : ",test_mask.shape)
    
    # sliding window
    batchSize,channel, depth, height, width = test_img.shape
    # volume_size = test_img.shape # (275, 512, 512)
    volume_size=[depth,height, width]
    # volume_size=np.asarray(volume_size)
    overlap_depth, overlap_height, overlap_width = overlap_size
    d_patch, h_patch, w_patch = patch_size
    # patch_width, patch_height, patch_depth = patch_size

    img_patches = []
    pred_patches = []
    mask_patches = []
    count=0

    for d in range(0, depth - d_patch + 1, d_patch - overlap_depth):
        for h in range(0, height - h_patch + 1, h_patch - overlap_height):
            for w in range(0, width - w_patch + 1, w_patch - overlap_width):
                
                
                img_patch = test_img[:,:, d:d + d_patch, h:h + h_patch, w:w + w_patch]
                # patch = volume[x:x+patch_width, y:y+patch_height, z:z+patch_depth]
                mask_patch= test_mask[:,:,d:d + d_patch, h:h + h_patch, w:w + w_patch]
                # print("img_patch shape: ",img_patch.shape) #torch.Size([96, 96, 96])      [1, 1, 96, 96, 96])

                img_patch2=img_patch.squeeze(0)
                img_patch2=img_patch2.squeeze(0)
                img_patch_npy=img_patch2.numpy()
                mask_patch2=mask_patch.squeeze(0)
                mask_patch2=mask_patch2.squeeze(0)
                mask_patch_npy=mask_patch2.numpy()
                # mask_patch_npy=mask_patch.detach().cpu().numpy()

                img_patch = img_patch.to(device)

                pred_patch= net(img_patch)
                mask_patch=mask_patch.to(torch.long)

                pred_patch = torch.argmax(pred_patch, dim=1) 
                pred_patch=pred_patch.squeeze(0)     
                pred_patch=pred_patch.detach().cpu()
                pred_patch_npy=pred_patch.numpy()

                pred_patches.append(pred_patch_npy)  
                img_patches.append(img_patch_npy)
                mask_patches.append(mask_patch_npy)

    pred_patches = np.asarray(pred_patches)
    # max_v= np.max(pred_patches)
    img_patches = np.asarray(img_patches)
    mask_patches = np.asarray(mask_patches)
    print('pred_patches shape:', pred_patches.shape)          # (147, 96, 96, 96)  
    print('img_patches shape:', img_patches.shape)   #   
    print('mask_patches shape:', mask_patches.shape)   #   

    pred_merged_volume = merge_patches(pred_patches, volume_size, overlap_size)  # 
    threshold = filters.threshold_otsu(pred_merged_volume)
    pred_vessels_bin = np.zeros(pred_merged_volume.shape)
    pred_vessels_bin[pred_merged_volume >= threshold] = 1
    pred_merged_volume=pred_vessels_bin

    print("pred_merged_volume   max value :",np.max(pred_merged_volume))
    print('pred_merged_volume shape:', pred_merged_volume.shape)  # (275, 512, 512)
    img_merged_volume = merge_patches(img_patches, volume_size, overlap_size)
    print('img_merged_volume shape:', img_merged_volume.shape)  # (275, 512, 512)
    mask_merged_volume = merge_patches(mask_patches, volume_size, overlap_size)
    # mask_merged_volume=np.uint8(mask_merged_volume)
    print('mask_merged_volume shape:', mask_merged_volume.shape)  # (275, 512, 512)

    intersection=(pred_merged_volume * mask_merged_volume).sum()
    Dice= (2*intersection)/(pred_merged_volume.sum()+mask_merged_volume.sum())
    Dice_mean.append(Dice)
    # Dice= np.sum(pred_merged_volume[mask_merged_volume == 1])*2.0/ (np.sum(pred_merged_volume)+ np.sum(mask_merged_volume))
    print("-------------------------Dice : ", Dice) # 

    pred_merged_volume = torch.from_numpy(pred_merged_volume).float()   # torch.Size([16, 1, 96, 96, 96])
    print("pred_merged_volume  shape : ", pred_merged_volume.shape)
    pred_merged_volume= pred_merged_volume.unsqueeze(0)
    # pred_merged_volume= pred_merged_volume.unsqueeze(0)
    print("pred_merged_volume  shape : ", pred_merged_volume.shape)
    mask_merged_volume = torch.from_numpy(mask_merged_volume).float()
    mask_merged_volume= mask_merged_volume.unsqueeze(0)
    # mask_merged_volume= mask_merged_volume.unsqueeze(0)
    tp, fn, fp, dice = metrics3d(pred_merged_volume, mask_merged_volume, pred_merged_volume.shape[0])  # pred.shape[0]=batch_size
    print(tp, fn, fp, dice)
    Recall=tp/(tp+fn + 1e-10)
    Precision=tp/(tp+fp + 1e-10)
    print("-------------------------Recall : ", Recall)
    print("-------------------------Precision : ", Precision)
    print("-------------------------dice222 : ", dice) # 
    Recall_mean.append(Recall)
    Precision_mean.append(Precision)



Dice_mean.sort(reverse=True)
average = sum(Dice_mean) / len(Dice_mean)
print("average dice:", average)
Recall_mean.sort(reverse=True)
average_Recall_mean = sum(Recall_mean) / len(Recall_mean)
print("average_Recall_mean :", average_Recall_mean)
Precision_mean.sort(reverse=True)
average_Precision_mean = sum(Precision_mean) / len(Precision_mean)
print("average_Precision_mean :", average_Precision_mean)


    