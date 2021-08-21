from efficientNet import Model
from dataloader import ShanghaiTech
from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models,transforms
import matplotlib.pyplot as plt

import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
import pdb
def test_model(model,optimizer,phase):
    since = time.time()
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    # Iterate over data.
    for index, (inputs, labels,segs, crowdcount) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        segs=segs.to(device)
        crowdcount=crowdcount.to(device)
        inputs = inputs.float()
        labels = labels.float()
        segs=segs.float()
        crowdcount=crowdcount.float()
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        labels = labels.view(-1,1,labels.shape[3],labels.shape[4])
        segs = segs.view(-1,1,segs.shape[3],segs.shape[4])
        #print("crowdcount.shape:", crowdcount.shape)
        crowdcount=labels.view(-1, 1)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(False):
            segs1,count1,labels1 = model(inputs)
            labels1 = labels1.to(torch.device("cpu")).numpy()/100
            pred_count = labels1.sum()
        
        true_count = crowdcount.sum()
        # print("true_count.shape:", true_count.shape)
        # print("true_count.item():", true_count.item())
        # print("true_count.item().shape", true_count.item().shape)

        #print(pred_count,true_count) 

        # backward + optimize only if in training phase
        mse = mse + torch.square(pred_count-true_count)
        mae = mae + torch.abs(pred_count-true_count)
        mre = mre + torch.abs(pred_count-true_count)/true_count
        pred[index,0] = pred_count
        pred[index,1] = true_count
    pred = pred[0:index+1,:]
    mse = torch.sqrt(mse/(index+1))
    mae = mae/(index+1)
    mre = mre/(index+1)
    print(phase+':')
    print(f'MAE:{mae:2.2f}, RMSE:{mse:2.2f}, MRE:{mre:2.4f}')
    time_elapsed = time.time() - since
    return pred,mae,mse,mre