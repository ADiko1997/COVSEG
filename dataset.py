import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import os 
import cv2
import torch
import random
import math

np.random.seed(0)
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)
#Data loader class

# +
class COVIDDataSet(Dataset):
    def __init__(self, root,  img_transform=None, mode=None):
        self.root = root
        self.img_transform = img_transform
        
        self.data_dir = root
        self.mask_dir = root+"_labels"
        self.img_list = os.listdir(self.data_dir)
        self.mask_list = os.listdir(self.mask_dir)
        self.mode = mode
 
    def __len__(self):
        return len(self.img_list)

    def backgElim(self, img, mask):

        background = (mask > 0.5)*1
        new_image = np.multiply(img, background)

        return new_image


    def transform(self, image, mask, mode=None):
        

        # Random crop

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.permute(2,0,1)
        mask = mask.view(1,512,512)

        if mode == 'train':
          if random.random() > 0.5:
              i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256,256))
              image = TF.crop(image, i, j, h, w)
              mask = TF.crop(mask, i, j, h, w)


          # Resize
          resize = transforms.Resize(size=(512, 512))
          image = resize(image)
          mask = resize(mask)

          # Random horizontal flipping
          if random.random() > 0.5:
              image = TF.hflip(image)
              mask = TF.hflip(mask)

          # Random vertical flipping
          if random.random() > 0.5:
              image = TF.vflip(image)
              mask = TF.vflip(mask)

        image = image.float()
        image = TF.normalize(image,mean=[0.2652102414756504, 0.2652102414756504, 0.2652102414756504],
                             std=[0.19156987584644022, 0.19156987584644022, 0.19156987584644022])

        return image, mask
 
    def __getitem__(self, index):
 
        if self.img_transform == None:
          imgs = cv2.imread(os.path.join(self.data_dir, self.img_list[index]))

          if '.jpeg' in self.img_list[index]:
            new_index =  self.mask_list.index( self.img_list[index])
          else:
            new_index =  self.mask_list.index( self.img_list[index][:-4]+".png")

          labels = cv2.imread(os.path.join(self.mask_dir, self.mask_list[new_index]))
          imgs = self.backgElim(imgs,labels)


          if '.jpeg' in self.img_list[index]:
            labels = np.where(labels == 2,1,labels)
            labels = np.where(labels > 2,2,labels)
          else:
            # labels = np.where(labels == 2,1,labels)
            labels = np.where(labels > 2,2,labels)

          labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)

          
          im =os.path.join(self.data_dir, self.img_list[index])
          mask = os.path.join(self.mask_dir, self.mask_list[new_index])
 
        else:
          imgs = cv2.imread(os.path.join(self.data_dir, self.img_list[index]))

          if '.jpeg' in self.img_list[index]:
            new_index =  self.mask_list.index( self.img_list[index])
          else:
            new_index =  self.mask_list.index( self.img_list[index][:-4]+".png")          
          
          labels = cv2.imread(os.path.join(self.mask_dir, self.mask_list[new_index]))
          imgs = self.backgElim(imgs,labels)

          if '.jpeg' in self.img_list[index]:
            labels = np.where(labels == 2,1,labels)
            labels = np.where(labels > 2,2,labels)
          else:
            # labels = np.where(labels == 2,1,labels)
            labels = np.where(labels > 2,2,labels)
#
          labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)

          im =os.path.join(self.data_dir, self.img_list[index])
          mask = os.path.join(self.mask_dir, self.mask_list[new_index])
          imgs, labels = self.transform(imgs,labels, self.mode)
 
        return imgs, labels, im, mask

class COVIDDataSetTest(Dataset):

    def __init__(self, root, bkg_mask_dir, img_transform=True):
        self.root = root
        self.img_transform = img_transform

        self.data_dir = root
        self.img_list = os.listdir(self.data_dir)

        self.bkg_mask_dir = bkg_mask_dir
        self.bkg_mask_list = os.listdir(self.bkg_mask_dir)
 
    def __len__(self):
        return len(self.img_list)


    def backgElim(self, img, mask):

        background = (mask > 0.5)*1
        new_image = np.multiply(img, background)

        return new_image


    def transform(self,image,mask):

        image = torch.from_numpy(image)
        image = image.permute(2,0,1)

        resize = transforms.Resize(size=(512, 512))
        image = resize(image)

        image = image.float()

        image = TF.normalize(image,mean=[0.2652102414756504, 0.2652102414756504, 0.2652102414756504],
                             std=[0.19156987584644022, 0.19156987584644022, 0.19156987584644022])

        return image

 
    def __getitem__(self, index):

        imgs = cv2.imread(os.path.join(self.data_dir, self.img_list[index]))
        new_index =  self.bkg_mask_list.index( self.img_list[index][:-5]+".jpeg")
        labels = cv2.imread(os.path.join(self.bkg_mask_dir, self.bkg_mask_list[new_index]))
        labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)

        im = self.img_list[index]
        imgs = self.transform(imgs,labels)    

        return imgs, im

# +
def WeightInit(path:str):

    list_os_train_masks = os.listdir(os.path.join(path,"Zenodo_train_labels"))
 
    counter_0 =0
    counter_1 =0
    counter_2 =0
    counter_3 =0
    
    for i in list_os_train_masks:
        image = cv2.imread(os.path.join(path,"Zenodo_train_labels",i))
        image = np.asarray(image)
        image = np.where(image==2,1,image)
        image = np.where(image>2,2,image)
        values = np.unique(image, return_counts=True)
    
    
        if len(values[0]) == 1:
            counter_0 += values[1][0]
        
        if len(values[0]) == 2:
            counter_0 += values[1][0]
            counter_1 += values[1][1]
    
        if len(values[0]) == 3:
            counter_0 += values[1][0]
            counter_1 += values[1][1]
            counter_2 += values[1][2]
    
#         if len(values[0]) == 4:
#             counter_0 += values[1][0]
#             counter_1 += values[1][1]
#             counter_2 += values[1][2]
#             counter_3 += values[1][3]
    
    
    num_pixels = 512 * 512 * len(list_os_train_masks) * 3
    rate_0 = counter_0/num_pixels
    rate_1 = counter_1/num_pixels
    rate_2 = counter_2/num_pixels
#     rate_3 = counter_3/num_pixels
    
    
    
    print("Rate 0:",rate_0)
    print("Rate 1:",rate_1)
    print("Rate 2:",rate_2)
#     print("Rate 3:",rate_3)
    
#     return  [1-rate_0, 1 - rate_1, 1-rate_2, 1-rate_3]
    return  [(1-rate_0)/3, (1 - rate_1)/3, (1-rate_2)/3]
#     return  [1,1,1]

