#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:22:13 2019

@author: fenqiang
"""


import torch
import torch.nn as nn
from torch.nn import init

import torchvision
import scipy.io as sio 
import numpy as np
import glob
import os
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
        

cuda = torch.device('cuda:1')
learning_rate = 0.000005
momentum = 0.99
weight_decay = 0.0001
batch_size = 4

train_dataset = ReconDataset(data_in, data_out, True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = ReconDataset(data_in, data_out, False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


#    dataiter = iter(train_dataloader)
#    image, gt = dataiter.next()

model = UNet(n_channels=10, n_classes=1)

print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

def get_learning_rate(epoch):
    limits = [2, 6, 8]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


def val_during_training(dataloader):
    model.eval()

    mae = torch.FloatTensor([0])
    for batch_idx, (image, gt) in enumerate(dataloader):
        image, gt = image.cuda(cuda), gt.cuda(cuda)
        with torch.no_grad():
            pred = model(image).squeeze()
        mae = mae + torch.abs(pred - gt).cpu().mean()
        
    mae = mae/float(len(dataloader))
    mae_m, mae_s = mae.mean(), mae.std()

    return  mae_m, mae_s


for epoch in range(300):
    
 #   mae_m, mae_s = val_during_training(train_dataloader)
  #  print("Train: mae_m, mae_s: {:.3} {:.3}".format(mae_m, mae_s))
#    writer.add_scalars('data/mean', {'train': m_mae}, epoch)
  #  mae_m, mae_s = val_during_training(val_dataloader)
   # print("Val: mae_m, mae_s: {:.4} {:.4}".format(mae_m, mae_s))    
#    writer.add_scalars('data/mean', {'val': m_mae}, epoch)

    lr = get_learning_rate(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))
    
    for batch_idx, (image, gt) in enumerate(train_dataloader):

        model.train()
        image = image.cuda(cuda)
        gt = gt.cuda(cuda)
    
        pred = model(image)
        pred = pred.squeeze()
       
        loss = ((pred-gt).abs() * (gt + 0.0016)).sum()/(batch_size*512*512.0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
 
    
    torch.save(model.state_dict(), "/home/fenqiang/baobao/model.pkl")

