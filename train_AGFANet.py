
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
from model.AGFANet import AGFANet
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
from torch.utils.data import ConcatDataset, DataLoader

batch_size = 1
batch_size_new=12  
num_epochs=500   
patch_size = [128, 160, 160] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
torch.cuda.empty_cache()
args = {
    # 'data_path': 'cta_project/data/npy',        
    'epochs': 10,
    'input_shape': (128, 160, 160),
    'snapshot': 2,  # 50
    'test_step': 2,
    'model_path': '/home/lxy/lxy/001_CASnet/save_models_randomcrop',
    'batch_size': 1,  # VNet 1 other 2
    'folder': 'folder2',
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
from torch.utils.data import DataLoader

#  load dataset
loaded_dataset1 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds002_1.pth')
print("Loaded loaded_dataset Size111:", len(loaded_dataset1))    # 
loaded_dataset_small1 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds0003_middleThresh1.pth')
concat_dataset1 = ConcatDataset([loaded_dataset1, loaded_dataset_small1])

loaded_dataset2 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds002_2.pth')
print("Loaded loaded_dataset Size222:", len(loaded_dataset2))    # 
concat_dataset22 = ConcatDataset([concat_dataset1, loaded_dataset2])

loaded_dataset3 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds002_3.pth')
print("Loaded loaded_dataset Size333:", len(loaded_dataset3))    # 
concat_dataset33 = ConcatDataset([concat_dataset22, loaded_dataset3])
# print("Loaded loaded_dataset Size:", len(concat_dataset3))    # 

loaded_dataset4 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds002_4.pth')
concat_dataset44 = ConcatDataset([concat_dataset33, loaded_dataset4])

loaded_dataset_small2 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds0003_middleThresh2.pth')
concat_dataset2 = ConcatDataset([concat_dataset44, loaded_dataset_small2])
loaded_dataset_small3_1 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds0003_smallThresh1.pth')
concat_dataset3_1 = ConcatDataset([concat_dataset2, loaded_dataset_small3_1])
loaded_dataset_small3_2 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds0003_smallThresh2.pth')
concat_dataset3_2 = ConcatDataset([concat_dataset3_1, loaded_dataset_small3_2])

train_dataset = concat_dataset3_2

loaded_dataset5 = torch.load('/home/lxy/lxy/data_CCTA/dataset128/new128_ds002_5.pth')
print("Loaded loaded_dataset Size555:", len(loaded_dataset5))    # 
validation_dataset = loaded_dataset5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size_new, shuffle=True) # here can be True
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size_new, shuffle=True)


# build model
Test_Model = {'AGFANet': AGFANet,
              'UNet3D': UNet3D
              }
net = Test_Model[args['model_name']](2, 1).to(device)
net = nn.DataParallel(net)

ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])
modelname = ckpt_path + '/' + '2024-05-02-4.pkl'

checkpoint = torch.load(modelname)
net.load_state_dict(checkpoint)

print("------------------------------------------")
num_para = 0
for name, param in net.named_parameters():
    num_mul = 1
    for x in param.size():
        num_mul *= x
    num_para += num_mul
# # print(net)
print("Number of trainable parameters {0} in Model {1}".format(num_para, str(args['model_name'])))  # 5647874   5962852
print("------------------------------------------")


critrion2 = WeightedCrossEntropyLoss().to(device)
critrion = nn.CrossEntropyLoss().to(device)
critrion3 = DiceLoss().to(device)
# Start training
print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
optimizer = torch.optim.Adam(net.parameters(), lr=1*1e-3, weight_decay=1e-8)   
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
iters = 1

def model_eval(net):
    print("\033[1;30;43m {} Start evaluation ... {}\033[0m".format("*" * 8, "*" * 8))   
    val_data_iter = iter(val_dataloader)
    
    for batch_idx in range(len(val_dataloader)):
        val_img, val_mask = next(val_data_iter)
        val_img = val_img.to(device)
        val_mask = val_mask.to(device)
        
        
        for m in net.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm3d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None
        
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.track_running_stats=False
                
        count = 0
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                count += 1
                if count >= 2:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
   
        TP, FN, FP, Dice = [], [], [], []
        pred_val = net(val_img)
        # loss_dice = critrion3(pred, train_mask)
        val_mask = val_mask.squeeze(1)
        val_mask=val_mask.to(torch.long)
        # label = label.squeeze(1)
        pred_val = torch.argmax(pred_val, dim=1)
        tp, fn, fp, dice = metrics3d(pred_val, val_mask, pred_val.shape[0])
        tp, fn, fp, dice = tp/ pred_val.shape[0], fn/ pred_val.shape[0], fp/ pred_val.shape[0], dice/ pred_val.shape[0]
        
        print(
                "--- test TP:{0:.4f}    test FN:{1:.4f}    test FP:{2:.4f}    test dice:{3:.4f}".format(tp, fn, fp, dice))
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)
        Dice.append(dice)

    return np.mean(TP), np.mean(FN), np.mean(FP), np.mean(Dice)

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    net.train()
    new_data_iter = iter(train_dataloader)
    for batch_idx in range(len(train_dataloader)):
            # print("batch_idx : ",batch_idx)
            optimizer.zero_grad()
            scheduler.step()
            train_img, train_mask = next(new_data_iter)
            train_img = train_img.to(device)
            train_mask = train_mask.to(device)
            pred = net(train_img)
    
            loss_dice = critrion3(pred, train_mask)  #0.477
            train_mask = train_mask.squeeze(1)
            train_mask=train_mask.to(torch.long)
            loss_ce = critrion(pred, train_mask)  #0.3145
            loss_wce = critrion2(pred, train_mask)  #0.3145
            loss = (loss_ce + 0.6 * loss_wce + 0.4 * loss_dice) / 3              # [1, 0.6, 0.4]

            loss_dice.backward()  # loss_dice   loss_ce  loss_wce
            optimizer.step()
            pred = torch.argmax(pred, dim=1)
            tp, fn, fp, dice = metrics3d(pred, train_mask, pred.shape[0])  # pred.shape[0]=batch_size
            if (epoch % 2) == 0:
                print(
                    '\033[1;36m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tdice:{6:.4f} '.format(
                                epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                                dice / pred.shape[0]))
            else:
                print(
                    '\033[1;32m [{0:d}:{1:d}] \u2501\u2501\u2501 loss:{2:.10f}\tTP:{3:.4f}\tFN:{4:.4f}\tFP:{5:.4f}\tdice:{6:.4f} '.format(
                        epoch + 1, iters, loss.item(), tp / pred.shape[0], fn / pred.shape[0], fp / pred.shape[0],
                        dice / pred.shape[0]))

            iters += 1

    if epoch >1:
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))


    # model eval
        
    if (epoch + 1) % args['test_step'] == 0:
        test_tp, test_fn, test_fp, test_dice = model_eval(net)
        print("Average TP:{0:.4f}, average FN:{1:.4f},  average FP:{2:.4f},  average Dice:{3:.4f}".format(test_tp,
                                                                                                             test_fn,
                                                                                                             test_fp,
                                                                                                             test_dice))
        if test_dice > max(best_score):
                best_score.append(test_dice)
                print("best_score: ",best_score)
                modelname = ckpt_path + '/' + 'best_score' + '_checkpoint.pkl'
                print('the best model will be saved at {}'.format(modelname))
                torch.save(net.state_dict(), modelname)
    print("------------------the best score of model is--------------- :", best_score)


    



    

















