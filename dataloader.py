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
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    d = os.path.join(dir,'images')
    for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    image_path = os.path.join(root, fname)
                    head,tail = os.path.split(root)
                    label_path = os.path.join(head,'ground-truth','GT_'+fname[:-4]+'.mat')
                    item = [image_path, label_path]
                    images.append(item)

    return images

IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train',extensions=IMG_EXTENSIONS,patch_size=128,num_patches_per_image=4):
        self.samples = make_dataset(data_dir,extensions)
        self.image_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.patch_size = patch_size
        self.numPatches = num_patches_per_image
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        img_file,label_file = self.samples[idx]
        image = cv2.imread(img_file)
        #kernel = np.array(1/16*[[1,2,1], [2,4,2], [1,2,1]])
        #image = cv2.filter2D(image, -1, kernel)
        height, width, channel = image.shape
        annPoints = scipy.io.loadmat(label_file)
        annPoints = annPoints['image_info'][0][0][0][0][0]
        positions = generate_density_map(shape=image.shape,points=annPoints,f_sz=15,sigma=4)
        fbs = generate_density_map(shape=image.shape,points=annPoints,f_sz=25,sigma=1)
        fbs = np.int32(fbs>0)
        targetSize = [self.patch_size,self.patch_size]
        height, width, channel = image.shape
        
        if (height < targetSize[0] or width < targetSize[1]):
            image = cv2.resize(image,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count = positions.sum()
            max_value = positions.max()
            # down density map
            positions = cv2.resize(positions, (np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            count2 = positions.sum()
            positions = np.minimum(positions*count/(count2+1e-8),max_value*10)
            #fbs = cv2.resize(fbs,(np.maximum(targetSize[0]+2,height),np.maximum(targetSize[1]+2,width)))
            #fbs = np.int32(fbs>0)
        
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        # transpose from h x w x channel to channel x h x w
        image = image.transpose(2,0,1)
        numPatches = self.numPatches
        if self.phase == 'train':
            patchSet, countSet,fbsSet, crowdcount = getRandomPatchesFromImage(image,positions,fbs,targetSize,numPatches)
            x = np.zeros((patchSet.shape[0],3,targetSize[0],targetSize[1]))
            if self.transform:
              for i in range(patchSet.shape[0]):
                #transpose to original:h x w x channel
                x[i,:,:,:] = self.transform(np.uint8(patchSet[i,:,:,:]).transpose(1,2,0))
            patchSet = x
        if self.phase == 'val' or self.phase == 'test':
            patchSet, countSet,fbsSet, crowdcount = getRandomPatchesFromImage(image, positions, fbs,targetSize,numPatches)
            patchSet[0,:,:,:] = self.transform(np.uint8(patchSet[0,:,:,:]).transpose(1,2,0))
        return patchSet, countSet, fbsSet,crowdcount

def getRandomPatchesFromImage(image,positions,fbs,target_size,numPatches):
    # generate random cropped patches with pre-defined size, e.g., 224x224
    imageShape = image.shape
    if np.random.random()>0.5:
        for channel in range(3):
            image[channel,:,:] = np.fliplr(image[channel,:,:])
        positions = np.fliplr(positions)
    patchSet = np.zeros((numPatches,3,target_size[0],target_size[1]))
    # generate density map
    countSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    fbsSet = np.zeros((numPatches,1,target_size[0],target_size[1]))
    crowdcount=np.zeros((numPatches,1))
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1]-target_size[0]+1)#x-height
        topLeftY = np.random.randint(imageShape[2]-target_size[1]+1)#y-width
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        patchSet[i,:,:,:] = thisPatch
        # density map
        position = positions[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        fb = fbs[topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        position = position.reshape((1, position.shape[0], position.shape[1]))
        fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
        crowdcount[i,:]=position.sum()
        countSet[i,:,:,:] = position
        fbsSet[i,:,:,:] = fb
    return patchSet, countSet,fbsSet, crowdcount

def getAllPatchesFromImage(image,positions,target_size):
    # generate all patches from an image for prediction
    nchannel,height,width = image.shape
    nRow = np.int(height/target_size[1])
    nCol = np.int(width/target_size[0])
    target_size[1] = np.int(height/nRow)
    target_size[0] = np.int(width/nCol)
    patchSet = np.zeros((nRow*nCol,3,target_size[1],target_size[0]))
    for i in range(nRow):
      for j in range(nCol):
        patchSet[i*nCol+j,:,:,:] = image[:,i*target_size[1]:(i+1)*target_size[1], j*target_size[0]:(j+1)*target_size[0]]
    return patchSet

def getAllFromImage(image,positions,fbs):
    nchannel, height, width = image.shape
    patchSet =np.zeros((1,3,224, 224))
    image=cv2.resize(image,(224,224))
    positions=cv2.resize(positions,(224,224))

    patchSet[0,:,:,:] = image[:,:,:]

    countSet = positions.reshape((1,1,positions.shape[0], positions.shape[1]))
    #fbsSet = fbs.reshape((1,1,fbs.shape[0], fbs.shape[1]))
    crowdcount=positions.sum()
    return patchSet, countSet,  crowdcount
