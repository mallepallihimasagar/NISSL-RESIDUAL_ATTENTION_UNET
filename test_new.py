import numpy as np
import torch
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import argparse
import copy
import os

from nissl_dataset import Nissl_Dataset
from network import U_Net,ResAttU_Net
#from loss_functions import DiceLoss
from metrics import dice_metric,pixel_accuracy,IoU,IoU_singlecell,calc_metrics
from skimage import io

torch.manual_seed(0)
np.random.seed(0)

#--------------------argparse arguemnts-----------------#
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,default='res_att_unet')
parser.add_argument('--trainset_path',type=str,default='Nissl_Dataset/train')
parser.add_argument('--testset_path',type=str,default='Nissl_Dataset/test')
parser.add_argument('--output',type=str,default='resattunet_output')
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--num_epochs',type=int,default=100)
parser.add_argument('--input_channels',type=int,default=3)
parser.add_argument('--output_channels',type=int,default=4)
parser.add_argument('--model_save_path',type=str,default='res_att_unet.pt')
parser.add_argument('--weights',type=str)

args = parser.parse_args()
# ------------------------parameters--------------------#
BATCH_SIZE = args.batch_size
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.1
MODEL_NAME = args.model_name

INPUT_CHANNELS=args.input_channels
OUTPUT_CHANNELS= args.output_channels
MODEL_PATH = args.weights #'/content/drive/MyDrive/final_models/trans_unet.pt'


OUTPUT_FOLDER = args.output

#dataset
train = Nissl_Dataset(root_dir='Nissl_Dataset/train',Transforms=False)
test = Nissl_Dataset(root_dir='Nissl_Dataset/test',Transforms=False)

#Dataloaders

val_size = int(train.__len__()*VALID_SPLIT)
train_size = train.__len__()-val_size
train_set,val_set = random_split(dataset=train,lengths=[train_size,val_size])

train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,pin_memory=True)
val_loader = DataLoader(dataset=val_set,batch_size=BATCH_SIZE,pin_memory=True)
test_loader = DataLoader(dataset=test,batch_size=BATCH_SIZE,pin_memory=True)

#DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'TESTING on {device}')

#get model
def get_model(model_name=None):
    model = None
    if model_name is None:
        print('Undefined model - Model Name = None')
        return None
    elif model_name == 'unet':
        model = U_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)
    elif model_name == 'res_att_unet':
        model = ResAttU_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)

    return model


model = get_model(model_name=MODEL_NAME).to(device)
model.load_state_dict(torch.load(MODEL_PATH), strict=False)


softmax = torch.nn.Softmax(dim=1)
os.mkdir(f'{OUTPUT_FOLDER}')

def rgb_output(image):
    colors = {'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255),'black':(0,0,0)}
    out = np.zeros((512,512,3))

    out[image==0,:]=colors['black']
    out[image==1,:]=colors['red']
    out[image==2,:]=colors['green']
    out[image==3,:]=colors['blue']

    return out

def masks_to_colorimg(masks):
    
    
    #colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228)])#,(56, 34, 132), (160, 194, 56)])

    colors = np.array([(0,0,0), (255,0,0), (0,255,0), (0,0,255)])
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            #selected_colors = colors[masks[:,y,x] > 0.5]
            #index = np.argmax(masks[:,y,x])
            index=0
            if masks[1,y,x]==1:
                index=1
            elif masks[2,y,x]==1:
                index=2
            elif masks[3,y,x]==1:
                index=3
            
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
dice_coef=[]
pixel_acc = []
iou1 = []
iou2=[]
iou3=[]

for idx,data in enumerate(test_loader):
    image,mask = data
    input = torch.true_divide(image,255).to(torch.float).to(device)

    target = mask.to(torch.long).to(device)


    output = model(input)

    final_output = output
    final_target = target

    d,p,i1,i2,i3 = calc_metrics(final_output.detach().cpu(),final_target.detach().cpu())

    dice_coef.append(d)
    pixel_acc.append(p)
    iou1.append(i1)
    iou2.append(i2)
    iou3.append(i3)

    
    output = torch.sigmoid(final_output).detach().cpu().squeeze(0).numpy()
    output = (output>0.5).astype(np.uint8)
    mask = final_target.detach().cpu()
    rgb_out = masks_to_colorimg(output)

    os.mkdir(f'{OUTPUT_FOLDER}/sample_{idx}')

    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/INPUT.png',image.squeeze(0).permute(1,2,0).numpy())
    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/GT.png',masks_to_colorimg(mask.squeeze(0).numpy()))
    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/pred.png',rgb_out)

print('Results on test set')
print(f'Dice coeff = {sum(dice_coef)/len(dice_coef)}')
print(f'pixel acc  = {sum(pixel_acc)/len(pixel_acc)}')
print(f'cell1 iou = {sum(iou1)/len(iou1)}')
print(f'cell2 iou = {sum(iou2)/len(iou2)}')
print(f'cell3 iou = {sum(iou3)/len(iou3)}')



