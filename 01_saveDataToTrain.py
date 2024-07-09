#  save_ckpt(net, str(epoch + 1)),  batch_size_new=16,     num_epochs=200  lr=1e-5

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
import pickle
import shutil
import os
import gzip
import nibabel as nib
import torch
from torch.utils.data import Dataset
import time
import argparse
import torch.distributed as dist
from model.csnet_3d import CSNet3D
from model.unet3d import UNet3D
# from dataloader.npy_3d_Loader import Data
from utils.train_metrics import metrics3d
from utils.losses import WeightedCrossEntropyLoss, DiceLoss
from torch.optim import lr_scheduler
import datetime
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import itk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("shuliang",torch.cuda.device_count())
print("device is " + str(device))
torch.cuda.empty_cache()
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
batch_size = 1
batch_size_new=64
num_epochs=200   #  600
patch_size = [128, 160, 160]    # [96, 96, 96]
args = {
    # 'data_path': 'cta_project/data/npy',
    'epochs': 10,
    'input_shape': (128, 160, 160),
    'snapshot': 50,
    'test_step': 1,
    'model_path': '/home/lxy/lxy/001_CASnet/save_models_randomcrop',
    'batch_size': 1,  # VNet 1 other 2
    'folder': 'folder3',
    'model_name': 'CSNet3D',  #UNet3D   CSNet3D
}

best_score = [0]
ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
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

        # 使用 nibabel 读取 .nii 文件
        # nii_data = nib.load(file_path).get_fdata()
        nii_data = itk.array_from_image(itk.imread(file_path))

        # 这里你可以根据需要进行进一步的数据处理，例如标准化、裁剪等
        if self.transform:
            nii_data = self.transform(nii_data)

        # 将数据转换为 PyTorch 的 Tensor 类型
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
from torch.utils.data import DataLoader

# 文件夹路径
data_folder = '/home/lxy/lxy/data_CCTA/train'
data_folder2 = '/home/lxy/lxy/data_CCTA/trainMask'
# data_folder = '/home/lxy/lxy/data_CCTA/train/patches_img'
# data_folder2 = '/home/lxy/lxy/data_CCTA/trainMask/patches_mask'
file_list_img = [f for f in os.listdir(data_folder) if f.endswith('.nii.gz')]
file_list_mask = [f for f in os.listdir(data_folder2) if f.endswith('.nii.gz')]
file_list_img=sorted(file_list_img)
file_list_mask=sorted(file_list_mask)
print('file_list_img  shape:', len(file_list_img))

train_size = int(len(file_list_img) * 1)    # ratio  of train vs validation.   0.8
train_img_list = file_list_img[:train_size]
train_mask_list = file_list_mask[:train_size]

# 创建自定义数据集实例，并传入标准化转换
dataset_img = CustomDataset(data_folder,train_img_list, transform=min_max_transform)
dataset_mask = CustomDataset(data_folder2,train_mask_list)

# dataloader
dataloader_img = DataLoader(dataset_img, batch_size=batch_size, shuffle=False)  # here must be False, follow the name
dataloader_mask = DataLoader(dataset_mask, batch_size=batch_size, shuffle=False)
print("len(dataloader_img :",len(dataloader_img))
print("len(dataloader_mask :",len(dataloader_mask))
iter_img = iter(dataloader_img)
iter_mask = iter(dataloader_mask)

Test_Model = {'CSNet3D': CSNet3D,
              'UNet3D': UNet3D
              }
def load_net():
    model = Test_Model[args['model_name']](2, 1).to(device)
    ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])
    # modelname = ckpt_path + '/' + 'best_score' + '_checkpoint300epo.pkl'
    modelname = ckpt_path + '/' + '2024-02-20-200.pkl'

    model = nn.DataParallel(model)  # 并行
    checkpoint = torch.load(modelname)
    # model.load_state_dict(checkpoint)

    return model

net = load_net()

critrion2 = WeightedCrossEntropyLoss().to(device)
critrion = nn.CrossEntropyLoss().to(device)
critrion3 = DiceLoss().to(device)
# Start training
print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-8)   # 第一次lr=1e-3    第二次 lr=1e-5
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)

new_data = []
threshold=0.002   # 0.015: 2/200     0.01: 19/200   0.012: 13/200
overlap_size=[32, 40, 40]


for batch_idx in range(len(dataloader_img)):
# for batch_idx in range(200):
            # print("batch_idx : ",batch_idx)
            net.train()
            train_img1 = next(iter_img)  # 1,1,275, 512, 512
            train_mask1 = next(iter_mask)

            batchSize,channel, depth, height, width = train_img1.shape
            volume_size=[depth,height, width]
            overlap_depth, overlap_height, overlap_width = overlap_size
            d_patch, h_patch, w_patch = patch_size

            img_patches = []
            pred_patches = []
            mask_patches = []
            count=0

            for d in range(0, depth - d_patch + 1, d_patch - overlap_depth):
                for h in range(0, height - h_patch + 1, h_patch - overlap_height):
                    for w in range(0, width - w_patch + 1, w_patch - overlap_width):
                        img_patch = train_img1[:,:, d:d + d_patch, h:h + h_patch, w:w + w_patch]
                        mask_patch= train_mask1[:,:, d:d + d_patch, h:h + h_patch, w:w + w_patch]
                        # img_patch_npy=img_patch.numpy()
                        img_patch = img_patch.to(device)
                        # mask_patch = mask_patch.to(device)

                        mask_patch1=mask_patch.squeeze(0)
                        mask_patch1=mask_patch1.squeeze(0)
                        mask_patch_npy=mask_patch1.numpy()
                        if np.sum(mask_patch_npy >0.5) < np.prod(mask_patch_npy.shape)* threshold:  # here is the mask
                            continue
                        new_data.append((train_img1, train_mask1))


            
print("len(new_data) : ",len(new_data))
new_dataset = TensorDataset(torch.cat([item[0] for item in new_data]),
                            torch.cat([item[1] for item in new_data]))
print("Loaded new_dataset Size:", len(new_dataset))
#  save 5  '.pth'  data
torch.save(new_dataset, '/home/lxy/lxy/data_CCTA/dataset/new128_dataset001.pth')  # new128_dataset001, new128_dataset002.....new128_dataset005


