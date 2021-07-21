import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


Height=512
Width = 512

class Nissl_Dataset(Dataset):
    def __init__(self,root_dir='Nissl_Dataset/train',Transforms=True,multiclass=True,cell_number=None):
        self.root_dir = root_dir
        self.transforms = Transforms
        self.multiclass = multiclass
        self.cell_number = cell_number

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def transform(self, image, mask):
        # Transform to tensor
        image = torch.from_numpy(image).permute(2,0,1)
        mask = torch.from_numpy(mask) #.unsqueeze(0)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        
        return image, mask

    def __getitem__(self,item):
        if torch.is_tensor(item):
            item = item.tolist()

        file_id = os.listdir(self.root_dir)[item]
        file_path = os.path.join(self.root_dir,file_id)
        

        cell1_mask = io.imread(file_path+f'/{file_id}_mask1.png')
        cell2_mask = io.imread(file_path+f'/{file_id}_mask2.png')
        cell3_mask = io.imread(file_path+f'/{file_id}_mask3.png')
        input_image = io.imread(file_path+f'/{file_id}.png')

        cell1_mask = cv2.resize(cell1_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        cell2_mask = cv2.resize(cell2_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        cell3_mask = cv2.resize(cell3_mask, (Width,Height),interpolation=cv2.INTER_NEAREST)
        input_image = cv2.resize(input_image,(Width,Height))
        
        
        mask = np.zeros((Width,Height),dtype=np.uint8)
        if self.multiclass:
            mask[cell1_mask==255]=1
            mask[cell2_mask==255]=2
            mask[cell3_mask==255]=3
        else :
            if self.cell_number==1:
                mask[cell1_mask==255]=1
            elif self.cell_number==2:
                mask[cell2_mask==255]=1
            elif self.cell_number==3:
                mask[cell3_mask==255]=1

        if self.transforms:

            input_image,mask = self.transform(input_image,mask)
        else:
            input_image = torch.from_numpy(input_image).permute(2,0,1)
            mask = torch.from_numpy(mask) #.unsqueeze(0)
            
        #converting 
        input_image = input_image.numpy()
        onehot_mask = torch.nn.functional.one_hot(mask.to(torch.long))
        onehot_mask = onehot_mask.permute(2,0,1).numpy()
        return input_image,np.expand_dims(onehot_mask[0],0)