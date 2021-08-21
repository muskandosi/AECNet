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
import torchvision.models as models
class model(nn.Module):
  def __init__(self):
    super(model, self).__init__()
    self.c=ContextualModule(2048,2048)
    self.model = EfficientNet.from_pretrained('efficientnet-b0')

    self.ps=nn.PixelShuffle(2)
    self.relu=nn.ReLU()
    self.tconv=nn.ConvTranspose2d(320,256,kernel_size=2,stride=2)

    self.tconv1=nn.ConvTranspose2d(256,256,kernel_size=2,stride=2)
    self.tconv2=nn.ConvTranspose2d(256,64,kernel_size=2,stride=2)
    self.mp=nn.MaxPool2d(2)
    self.tconv3=nn.ConvTranspose2d(64,1,kernel_size=4,stride=4)
    self.tconv4=nn.ConvTranspose2d(64,1,kernel_size=2,stride=2)
    self.conv=nn.Conv2d(1,1,kernel_size=1)
    self.sig=nn.Sigmoid()
    self.flat=nn.Flatten()
    self.reg=nn.Linear(62720,512)
    self.reg1=nn.Linear(512,64)
    self.reg2=nn.Linear(64,1)
    self.conv1=nn.Conv2d(64,64,kernel_size=1)
    self.conv2=nn.Conv2d(1,1,kernel_size=1)
  def forward(self,x):
    x = self.model.extract_features(x)
    #1280x7x7
    x=self.relu(x)
    x1=x
    x=self.ps(x)
    #320, 14, 14
    x=self.tconv(x)
    #256,28,28
    x=self.tconv1(x)
    x2=x
    #256,56,56
    x=self.tconv2(x)
    x=self.conv1(x)
    #64,112,112
    x=self.mp(x)
    #64,56,56
    x=self.tconv3(x)
    x=self.conv2(x)
    #1,224,224
    x=self.sig(x)

    x1=self.flat(x1)
    x1=self.reg(x1)
    x1=self.reg1(x1)
    x1=self.reg2(x1)
    x1=self.relu(x1)
    
    x2=self.tconv2(x2)
    #64,112,112
    x2=self.tconv4(x2)
    #1,224,224
    x2=x2*x
    x2=self.conv(x2)
    x2=self.conv(x2)
    x2=self.relu(x2)
    #1,224,224
    return x,x1,x2
