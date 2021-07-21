import numpy as np
import torch
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import os

from nissl_dataset import Nissl_Dataset
from network import U_Net,ResAttU_Net
#from loss_functions import DiceLoss
#from metrics import dice_metric,pixel_accuracy,IoU,IoU_singlecell
from skimage import io


torch.manual_seed(0)
np.random.seed(0)

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.1
MODEL_NAME = 'unet'

INPUT_CHANNELS=3
OUTPUT_CHANNELS=4 
MODEL_PATH = '/content/drive/MyDrive/final_models/unet.pt'
OUTPUT_FOLDER = 'unet_bce_dice'

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
    if model_name==None:
        print('Undefined model - Model Name = None')
        return None
    elif model_name == 'unet':
        model = U_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)
    elif model_name == 'res_att_unet':
        model = ResAttU_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)

    return model

model = get_model(model_name=MODEL_NAME)
model = model.to(device)

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


for idx,data in enumerate(val_loader):
                
    image,mask = data
    input = torch.true_divide(image,255).to(torch.float).to(device)
    target = mask.to(torch.long).to(device)
    output = model(input)

    output = softmax(output).squeeze(dim=0).detach().cpu()
    rgb_out = masks_to_colorimg(output.numpy())

    os.mkdir(f'{OUTPUT_FOLDER}/sample_{idx}')

    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/INPUT.png',image.squeeze(0).permute(1,2,0).numpy())
    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/GT.png',masks_to_colorimg(mask.squeeze(0).numpy()))
    io.imsave(f'{OUTPUT_FOLDER}/sample_{idx}/pred.png',rgb_out)
    if idx>4:
      break
    



