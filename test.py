import torch
import matplotlib.pyplot as plt
import numpy as np
from nissl_dataset import Nissl_mask_dataset

import time
from collections import defaultdict
import torch.nn.functional as F
import torch
from loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np


torch.manual_seed(0)
np.random.seed(0)

# ------------------------parameters--------------------#
batch_size = 4
# ------------------------dataset-----------------------#
dataset = Nissl_mask_dataset()
dataset_len = dataset.__len__()

# train, val, test = random_split(dataset, [dataset_len-60, 30, 30])
# noinspection PyArgumentList
train, val, test = random_split(dataset, [228, 32, 66])
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val, batch_size=batch_size // 2, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test, batch_size=batch_size // 4, shuffle=True, num_workers=4)


from nissl_dataset import Nissl_mask_dataset
from network import U_Net
from network import ResAttU_Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

modelunet = U_Net(UnetLayer=5, img_ch=3, output_ch=4).to(device)
modelresunet = ResAttU_Net(UnetLayer=5,img_ch=3,output_ch=4).to(device)
modelunet.load_state_dict(torch.load('/gdrive/MyDrive/models/unet'), strict=False)

# output = modelunet(image.to(device))

modelresunet.load_state_dict(torch.load('/gdrive/MyDrive/models/resunet'), strict=False)

def gt_to_colorimg(masks):
    
    
    #colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228)])#,(56, 34, 132), (160, 194, 56)])

    colors = np.asarray([(0,0,0), (255,0,0), (0,255,0), (0,0,255)])
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]
            
            if len(selected_colors) > 0:
              if masks[:,y,x][3]:
                colorimg[y,x,:]=colors[3]
              elif masks[:,y,x][2]:
                colorimg[y,x,:]=colors[2]
              elif masks[:,y,x][1]:
                colorimg[y,x,:]=colors[1]
              else :
                colorimg[y,x,:]=colors[0] 
              # colorimg[y,x,:] = np.mean(selected_colors, axis=0)


    return colorimg.astype(np.uint8)

def masks_to_colorimg(masks):
    
    
    #colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228)])#,(56, 34, 132), (160, 194, 56)])

    colors = np.asarray([(0,0,0), (255,0,0), (0,255,0), (0,0,255)])
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            #selected_colors = colors[masks[:,y,x] > 0.5]
            index = np.argmax(masks[:,y,x])
            
            
            if index==2:
              colorimg[y,x,:]=colors[2]
            elif index==1:
              colorimg[y,x,:]=colors[1]
            elif index==3:
              colorimg[y,x,:]=colors[3]
            else :
              colorimg[y,x,:]=colors[0] 
              # colorimg[y,x,:] = np.mean(selected_colors, axis=0)


    return colorimg.astype(np.uint8)

import matplotlib.pyplot as plt
import os
def make_binary(out,index):

  # plt.imshow(input.squeeze().cpu().permute(1,2,0))
  out = out.squeeze()[index]
  # out = torch.true_divide(out,255)
  
  out = out.cpu().numpy()
  out[out>=0.5] = 1
  out[out<0.5] = 0
  return out

unet_cell1_acc=[]
unet_cell2_acc=[]
unet_cell3_acc=[]

resunet_cell1_acc=[]
resunet_cell2_acc=[]
resunet_cell3_acc=[]
with torch.no_grad():
  count = 1
  try:
    os.mkdir('output')
  except:
    print("output directory exists \n saving samples to /output/")
  for input,target in test_loader:
    print("saving sample_{} to output/".format(count))
    input = torch.true_divide(input,255)
    input = input.to(device)
    os.mkdir('output/sample_{}'.format(count))
    input= input.type(torch.float)
    output = modelunet(input)
    output2 = modelresunet(input)

    plt.axis('off')
    plt.imshow(input.squeeze().cpu().permute(1,2,0))
    plt.savefig('output/sample_{}/input.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(target.squeeze().cpu()[1],cmap='gray')
    plt.savefig('output/sample_{}/gt_cell1.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(target.squeeze().cpu()[2],cmap='gray')
    plt.savefig('output/sample_{}/gt_cell2.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(target.squeeze().cpu()[3],cmap='gray')
    plt.savefig('output/sample_{}/gt_cell3.png'.format(count), bbox_inches='tight', pad_inches=0)

    #unet output
    plt.imshow(make_binary(output,1),cmap='gray')
    plt.savefig('output/sample_{}/unet_cell1.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(make_binary(output,2),cmap='gray')
    plt.savefig('output/sample_{}/unet_cell2.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(make_binary(output,3),cmap='gray')
    plt.savefig('output/sample_{}/unet_cell3.png'.format(count), bbox_inches='tight', pad_inches=0)

    target1 = target.squeeze().cpu()[1].numpy().reshape(512*512)
    unet1 = make_binary(output,1).reshape(512*512)
    unet_cell1_acc.append(sum(target1==unet1)/target1.shape[0])

    target2 = target.squeeze().cpu()[2].numpy().reshape(512*512)
    unet2 = make_binary(output,2).reshape(512*512)
    unet_cell2_acc.append(sum(target2==unet2)/target2.shape[0])

    target3 = target.squeeze().cpu()[3].numpy().reshape(512*512)
    unet3 = make_binary(output,3).reshape(512*512)
    unet_cell3_acc.append(sum(target3==unet3)/target3.shape[0])
    #resunet output
    plt.imshow(make_binary(output2,1),cmap='gray')
    plt.savefig('output/sample_{}/resunet_cell1.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(make_binary(output2,2),cmap='gray')
    plt.savefig('output/sample_{}/resunet_cell2.png'.format(count), bbox_inches='tight', pad_inches=0)
    plt.imshow(make_binary(output2,3),cmap='gray')
    plt.savefig('output/sample_{}/resunet_cell3.png'.format(count), bbox_inches='tight', pad_inches=0)

    #target1 = target.squeeze().cpu()[1].numpy().reshape(512*512)
    resunet1 = make_binary(output2,1).reshape(512*512)
    resunet_cell1_acc.append(sum(target1==resunet1)/target1.shape[0])

    #target2 = target.squeeze().cpu()[2].numpy().reshape(512*512)
    resunet2 = make_binary(output2,2).reshape(512*512)
    resunet_cell2_acc.append(sum(target2==resunet2)/target2.shape[0])

    #target3 = target.squeeze().cpu()[3].numpy().reshape(512*512)
    resunet3 = make_binary(output2,3).reshape(512*512)
    resunet_cell3_acc.append(sum(target3==resunet3)/target3.shape[0])


    target_rgb = gt_to_colorimg(target.squeeze().cpu().numpy())
    plt.imshow(target_rgb)
    plt.savefig('output/sample_{}/gt_rgb.png'.format(count), bbox_inches='tight', pad_inches=0)

    unet_pred_rgb = masks_to_colorimg(output.squeeze().cpu().numpy())
    plt.imshow(unet_pred_rgb)
    plt.savefig('output/sample_{}/unet_rgb.png'.format(count), bbox_inches='tight', pad_inches=0)

    resunet_pred_rgb = masks_to_colorimg(output2.squeeze().cpu().numpy())
    plt.imshow(resunet_pred_rgb)
    plt.savefig('output/sample_{}/resunet_rgb.png'.format(count), bbox_inches='tight', pad_inches=0)

    count+=1
    if count>=10:
      break

