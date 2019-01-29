#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:32:32 2019

@author: fenqiang
"""



import torch
import torch.nn as nn
from torch.nn import init

import torchvision
import scipy.io as sio 
import numpy as np
import imageio
import matplotlib.pyplot as plt

from unet import UNet
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

file = "/home/fenqiang/baobao/data/images_in1.mat"
raw = sio.loadmat(file)
raw = raw['I_all_in']
raw = raw[0]
data_in = np.zeros((250, 256, 256, 10))
for i in range(250):
    data_in[i,:,:,:] = raw[i]


file = "/home/fenqiang/baobao/data/images_out1.mat"
raw = sio.loadmat(file)
raw = raw['I_all_out']
raw = raw[0]
data_out = np.zeros((250, 256, 256))
for i in range(250):
    data_out[i,:,:] = raw[i]

data_in = np.swapaxes(data_in, 1,3)
data_in = np.swapaxes(data_in, 2,3)/255.0



class ReconDataset(torch.utils.data.Dataset):

    def __init__(self, data_in, data_out, train):
        self.train = train
        if train:
            self.data_in = data_in[0:200,:,:,:]    
            self.data_out = data_out[0:200,:,:]
        else:
            self.data_in = data_in[200:250,:,:,:]    
            self.data_out = data_out[200:250,:,:]
            
    def __getitem__(self, index):
        image = self.data_in[index,:,:,:]
        groundtruth = self.data_out[index,:,:]
        return image.astype(np.float32), groundtruth.astype(np.float32), index
        
    def __len__(self):
        if self.train:
            return 200
        else:
            return 50
        
        
batch_size = 1
cuda = torch.device('cuda:0')

val_dataset = ReconDataset(data_in, data_out, False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(val_dataloader)
#    image, gt, index = dataiter.next()

model = UNet(n_channels=10, n_classes=1)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
model.load_state_dict(torch.load("/home/fenqiang/baobao/model.pkl"))
model.eval()


for batch_idx, (image, gt, index) in enumerate(val_dataloader):

    image, gt = image.cuda(cuda), gt.cuda(cuda).squeeze()

    pred = model(image)
    pred = pred.squeeze()
   
    loss = ((pred-gt).abs() * (gt + 1.0)).sum()/(pred.shape[0] * pred.shape[1])
    
    print ("loss: ", loss)
    imageio.imwrite('prediction/' + str(index[0].numpy()) + '_gt.png', gt.cpu().numpy())
    imageio.imwrite('prediction/' + str(index[0].numpy()) + '_pred.png', pred.detach().cpu().numpy())
    