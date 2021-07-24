import time
import argparse
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

from nissl_dataset import Nissl_Dataset
from network import U_Net
from network import ResAttU_Net

torch.manual_seed(0)
np.random.seed(0)

#--------------------argparse arguemnts-----------------#
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,default='res_att_unet')
parser.add_argument('--trainset_path',type=str,default='Nissl_Dataset/train')
parser.add_argument('--testset_path',type=str,default='Nissl_Dataset/test')
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--num_epochs',type=int,default=100)
parser.add_argument('--input_channels',type=int,default=3)
parser.add_argument('--output_channels',type=int,default=4)
parser.add_argument('--model_save_path',type=str,default='res_att_unet.pt')
parser.add_argument('--weights',type=str)

args = parser.parse_args()


# ------------------------parameters--------------------#
batch_size = args.batch_size
NUM_EPOCHS = args.num_epochs
INPUT_CHANNELS = args.input_channels
OUTPUT_CHANNELS = args.output_channels
MODEL_NAME = args.model_name #'res_att_unet'
MODEL_SAVE_PATH = args.model_save_path #"res_att_unet.pt"
# ------------------------dataset-----------------------#
train_dataset = Nissl_Dataset(root_dir=args.trainset_path)
train_dataset_len = train_dataset.__len__()

test_dataset = Nissl_Dataset(root_dir=args.testset_path)
test_dataset_len = test_dataset.__len__()
# ------------------------creating model file-----------------------#
try:
    with open(MODEL_SAVE_PATH,'w') as fp:
        print(f'File created at {MODEL_SAVE_PATH}')
except:
    print(f'{MODEL_SAVE_PATH} file exits!')


print(f"training with {train_dataset_len} images")
# train, val, test = random_split(dataset, [dataset_len-60, 30, 30])
# noinspection PyArgumentList
train, val = random_split(train_dataset, [train_dataset_len - 30,30])
train_loader = DataLoader(dataset=train, batch_size=batch_size, num_workers=2)
val_loader = DataLoader(dataset=val, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test' : test_loader
}

inp,msk = next(iter(train_loader))
print("input and mask shapes",inp.shape,msk.shape)

# -----------------------training-----------------------#

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    pred_flat = pred.view(-1).data.cpu().numpy()
    target_flat =target.view(-1).data.cpu().numpy()

    # pred_flat[pred_flat>=0.5]=1
    # pred_flat[pred_flat < 0.5] = 0

    #pred_flat = pred_flat//255
    # acc = np.sum(pred_flat==target_flat)/pred_flat.shape[0]
    # f1score = f1_score(pred_flat,target_flat)
    #pixel_acc = torch.true_divide(torch.sum(pred_flat==target.view(-1)),pred.view(-1).shape[0])



    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    #metrics['pixel_acc'] = acc
    #metrics['f1_score'] = f1score

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = [
        "{}: {:4f}".format(k, metrics[k] / epoch_samples)
        for k in metrics.keys()
    ]
    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    patience = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = torch.true_divide(inputs, 255)
                inputs = inputs.type(torch.float)
                labels = labels.type(torch.float)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase== 'val':
                if epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience +=1


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if patience >= 15:
            print("out of patience breaking the training loop :(")
            break

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')


# -----------------------model--------------------------#
def get_model(model_name=None):
    if model_name is None:
        print('Undefined model - Model Name = None')
        return None
    elif model_name == 'unet':
        model = U_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)
    elif model_name == 'res_att_unet':
        model = ResAttU_Net(UnetLayer=5,img_ch=INPUT_CHANNELS,output_ch=OUTPUT_CHANNELS)

    return model

model = get_model(MODEL_NAME).to(device)

if args.weights:
    print("Loading pre-trained weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights), strict=False)

# freeze backbone layers
# Comment out to finetune further
# for l in model.base_layers:
#     for param in l.parameters():
#         param.requires_grad = False

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
try:
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved succesfully at {MODEL_SAVE_PATH}')
except:
    print(f'Failed to save the model at {MODEL_SAVE_PATH}')
