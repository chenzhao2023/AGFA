# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
import torch 
import numpy as np 
from PIL import Image
import glob 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import os
import nibabel as nib
import pickle
import shutil
import os
import gzip



def rename_files(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        new_filename = os.path.basename(subfolder) + 'img.nii.gz'
        new_filename2 = os.path.basename(subfolder) + 'label.nii.gz'
        
        old_filepath = os.path.join(subfolder, 'img.nii.gz')
        old_filepath2 = os.path.join(subfolder, 'label.nii.gz')

        new_filepath = os.path.join(subfolder, new_filename)
        new_filepath2 = os.path.join(subfolder, new_filename2)
        
        os.rename(old_filepath, new_filepath)
        os.rename(old_filepath2, new_filepath2)

        print(f'Renamed: {old_filepath2} -> {new_filepath2}')


def move_files_to_folderB(source_folder, destination_folder):
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    
    for subfolder in subfolders:
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]

        for file_path in files:
            dest_file_path = os.path.join(destination_folder, os.path.basename(file_path))
            
            shutil.move(file_path, dest_file_path)
            print(f'Moved: {file_path} -> {dest_file_path}')


def move_files_with_label(source_folder, destination_folder):
    
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    labeled_files = [f for f in files if 'label' in f]

    os.makedirs(destination_folder, exist_ok=True)

    for file_name in labeled_files:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        
        shutil.move(source_file_path, destination_file_path)
        print(f'Moved: {source_file_path} -> {destination_file_path}')



