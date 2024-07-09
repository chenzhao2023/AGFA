import numpy as np
import os
import itk
import nibabel as nib
import glob 
import matplotlib.pyplot as plt
import PIL
from PIL import Image

patch_size = [96, 96, 96]
overlap_size = [32, 32, 32]
class MinMaxScale:
    def __call__(self, img):
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val)
        return img
min_max_transform = MinMaxScale()
def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


# 文件夹路径
# folder_img = '/home/lxy/lxy/data_CCTA/train/patches_img'
# folder_mask = '/home/lxy/lxy/data_CCTA/trainMask/patches_mask'

folder_img = '/home/lxy/lxy/data_CCTA/train/patches_img'   # /patches_mask
folder_mask = '/home/lxy/lxy/data_CCTA/trainMask/patches_mask'   # /patches_mask
save_img = '/home/lxy/lxy/data_CCTA/train/imgs_1_nii'

file_list_img = [f for f in os.listdir(folder_img) if f.endswith('.nii.gz')]
file_list_mask = [f for f in os.listdir(folder_mask) if f.endswith('.nii.gz')]
file_list_img=sorted(file_list_img)
file_list_mask=sorted(file_list_mask)

# take the first
file_name = file_list_img[120]
file_nameMask = file_list_mask[120] 

file_path = os.path.join(folder_img, file_name)      # train2  是做实验。
file_pathMask = os.path.join(folder_mask, file_nameMask)

imgs = itk.array_from_image(itk.imread(file_path))
masks = itk.array_from_image(itk.imread(file_pathMask))

print("imgs shape: ", imgs.shape)
print("masks shape: ", masks.shape)

img=imgs[8,: ,:]  # imgs 从 -1024 到3071
normalized_img = normalize_array(img)   # 0-1
normalized_img= normalized_img * 255
img_pic= Image.fromarray(img)

mask=masks[8,: ,:]* 255 #  masks array 是 0-1的
print("sum mask: ", sum(sum(mask)))
mask_pic= Image.fromarray(mask)


plt.figure()
plt.tight_layout()
plt.imshow(img_pic, cmap='gray')
plt.show()
# plt.show(block=True)
plt.savefig('/home/lxy/lxy/data_CCTA/train/imgs_1_nii/img.jpg' )
            # plt.savefig('./result/pic_Cycle-{}.png'.format(epoch + 1))
plt.close()


plt.figure()
plt.tight_layout()
plt.imshow(mask_pic, cmap='gray')
plt.show()
# plt.show(block=True)
plt.savefig('/home/lxy/lxy/data_CCTA/trainMask/masks_1_nii/mask.jpg' )
            # plt.savefig('./result/pic_Cycle-{}.png'.format(epoch + 1))
plt.close()





















