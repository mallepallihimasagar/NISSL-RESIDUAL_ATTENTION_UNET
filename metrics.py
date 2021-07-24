import torch
import numpy as np

#accuracy metrics
def dice_metric(inputs, targets,from_logits=True):
    if from_logits:
        torch.sigmoid(inputs)
    #target should be one-hot encoding
    inputs = inputs.reshape(-1).numpy()
    targets = targets.reshape(-1).numpy()
    inputs = (inputs>0.5).astype(np.uint8)
    intersection = 2.0 * (targets * inputs).sum()
    union = targets.sum() + inputs.sum()
    if targets.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def pixel_accuracy(inputs,targets,from_logits=True):
    inputs = inputs.to(torch.float)
    targets = targets.to(torch.float)
    if from_logits:
        softmax = torch.nn.Softmax(dim=1)
        inputs = softmax(inputs)
        inputs = torch.argmax(inputs,dim=1)
        targets = softmax(targets)
        targets = torch.argmax(targets,dim=1)

    
    inputs = inputs.view(-1).numpy()
    targets = targets.view(-1).numpy()
    
    return (inputs==targets).sum()/targets.shape[0]

def IoU(inputs,targets,cell=None,from_logits=True):
    inputs = inputs.to(torch.long)
    targets = targets.to(torch.long)
    if from_logits:
        torch.sigmoid(inputs)
    #target should be one-hot encoding
    if cell:
        inputs = inputs[:,cell,:,:]
        targets = targets[:,cell,:,:]
    
    inputs = inputs.reshape(-1).numpy()
    targets = targets.view(-1).numpy()
    inputs = (inputs>0.5).astype(np.uint8)
    intersection = (inputs&targets).sum()
    #union = target.sum() + inputs.sum()
    union = (inputs|targets).sum()
    if targets.sum() == 0 and inputs.sum() == 0:
        return 1.0    
    return intersection/union

def IoU_singlecell(inputs,targets,from_logits=True):
    if from_logits:
        torch.sigmoid(inputs)
    #target should be one-hot encoding
    
    #inputs = inputs[:,1,:,:]
    #targets = targets[:,1,:,:]
    
    inputs = inputs.reshape(-1).numpy()
    targets = targets.reshape(-1).numpy()
    inputs = (inputs>0.5).astype(np.uint8)
    intersection = (inputs&targets).sum()
    #union = target.sum() + inputs.sum()
    union = (inputs|targets).sum()
    if targets.sum() == 0 and inputs.sum() == 0:
        return 1.0    
    return intersection/union

def calc_metrics(output,target):
    
    onehot_target = target#torch.nn.functional.one_hot(target.to(torch.long)).permute(0,3,1,2)
    dice_coeff = dice_metric(output,onehot_target,from_logits=True)
    pixel_acc = pixel_accuracy(output,target,from_logits=True)
    cell1_iou = IoU(output,onehot_target,cell=0,from_logits=True)
    cell2_iou = IoU(output,onehot_target,cell=1,from_logits=True)
    cell3_iou = IoU(output,onehot_target,cell=2,from_logits=True)

    # metrics = {"dice_coeff" :dice_coeff,
    #             "pixel_accuracy" : pixel_acc,
    #             "cell_123_IoU" : (cell1_iou,cell2_iou,cell3_iou)
    #             }

    return (dice_coeff,pixel_acc,cell1_iou,cell2_iou,cell3_iou)