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
seg_loss = True
test_step = 1
batch_size = 6
num_workers = 2
patch_size = 224
num_patches_per_image = 4
data_dir = '/content/drive/MyDrive/part_B/'

# define data set
image_datasets = {x: ShanghaiTechDataset(data_dir+x+'_data', 
                        phase=x, 
                        transform=data_transforms[x],
                        patch_size=patch_size,
                        num_patches_per_image=num_patches_per_image)
                    for x in ['train','test']}
image_datasets['val'] = ShanghaiTechDataset(data_dir+'train_data',
                            phase='val',
                            transform=data_transforms['val'],
                            patch_size=patch_size,
                            num_patches_per_image=num_patches_per_image)
## split the data into train/validation/test subsets
indices = list(range(len(image_datasets['train'])))
split = np.int(len(image_datasets['train'])*0)

val_idx = np.random.choice(indices, size=split, replace=False)
train_idx = indices#list(set(indices)-set(val_idx))
test_idx = range(len(image_datasets['test']))

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetSampler(test_idx)

train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'],batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'],batch_size=1,sampler=val_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'],batch_size=1,sampler=test_sampler, num_workers=num_workers)

dataset_sizes = {'train':len(train_idx),'val':len(val_idx),'test':len(image_datasets['test'])}
dataloaders = {'train':train_loader,'val':val_loader,'test':test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mymodel=model()
mymodel=mymodel.to(device)
optimizer = optim.Adam(mymodel.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

since = time.time()
best_model_wts = copy.deepcopy(mymodel.state_dict())
best_mae_val = 1e6
best_mae_by_val = 1e6
best_mae_by_test = 1e6
best_mse_by_val = 1e6
best_mse_by_test = 1e6
criterion1 = nn.MSELoss(reduce=False) # for density map loss
criterion2 = nn.MSELoss() # for count loss
criterion3 = nn.BCELoss() #segmentation map
for epoch in range(300):
  print('Epoch {}/{}'.format(epoch, 201 - 1))
  print('-' * 10)
  running_loss = 0.0
  for index, (inputs, labels,segs, crowdcount) in enumerate(dataloaders['train']):
    labels = labels*100
    labels = skimage.measure.block_reduce(labels.numpy(),(1,1,1,1,1),np.sum)
    labels = torch.from_numpy(labels)
    #crowdcount = torch.from_numpy(crowdcount)
    labels = labels.to(device)
    segs=segs.to(device)
    crowdcount = crowdcount.to(device)
    inputs = inputs.to(device)
    inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
    labels = labels.view(-1,1,labels.shape[3],labels.shape[4])

    segs = segs.view(-1,1,segs.shape[3],segs.shape[4])

    crowdcount = crowdcount.view(-1,1)
    inputs = inputs.float()
    labels = labels.float()
    segs=segs.float()
    crowdcount=crowdcount.float()
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      segs1,count1,labels1 = mymodel(inputs)
      loss_den = criterion1(labels1, labels)
      count=count1.sum()
      loss_count = criterion2(count, crowdcount)
      loss_seg=criterion3(segs1,segs)
      th=100 # no curriculum loss when th is set a big number
      weights = th/(F.relu(labels-th)+th)
      
      if seg_loss:
          loss = 100*loss_den +10*loss_seg+ 0.5*loss_count
      else:
          loss = loss_den
      loss = loss*weights
      loss = loss.sum()/weights.sum()
      loss.backward()
      optimizer.step()
    running_loss += loss.item() * inputs.size(0)
               
  #scheduler.step()
  exp_lr_scheduler.step()
  print("dataset_sizes:", dataset_sizes)
  epoch_loss = running_loss / dataset_sizes['train']            
        
  print('Train Loss: {:.4f}'.format(epoch_loss))
  if epoch%test_step==0:
    #tmp,epoch_mae,epoch_mse,epoch_mre=test_model(mymodel,optimizer,'val')
    tmp,epoch_mae_test,epoch_mse_test,epoch_mre_test = test_model(mymodel,optimizer,'test')
    '''
    if  epoch_mae < best_mae_val:
      best_mae_val = epoch_mae
      best_mae_by_val = epoch_mae_test
      best_mse_by_val = epoch_mse_test
      best_epoch_val = epoch
'''
    if epoch_mae_test < best_mae_by_test:
      best_mae_by_test = epoch_mae_test
      best_mse_by_test = epoch_mse_test
      best_epoch_test = epoch
    print()
    #print('best MAE and MSE by val:  {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_val,best_mse_by_val, best_epoch_val))
    print('best MAE and MSE by test: {:2.2f} and {:2.2f} at Epoch {}'.format(best_mae_by_test,best_mse_by_test, best_epoch_test))
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# load best model weights
mymodel.load_state_dict(best_model_wts)