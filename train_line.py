#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 18:36:01 2019

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


class ReconDataset(torch.utils.data.Dataset):

    def __init__(self, train = True):

        if train is True:
            self.files = np.arange(200) + 1 
        else:
            self.files = np.arange(50) + 201

    def __getitem__(self, index):
        file_in = "/media/fenqiang/DATA/baobao/simulation_data/lines_images_in_" + str(self.files[index]) + '.mat'
        data = sio.loadmat(file_in)
        data = data['I_ins']
        data = np.swapaxes(data, 0,2)
        data = np.swapaxes(data, 1,2)/255.0
        file_gt = "/media/fenqiang/DATA/baobao/simulation_data/" + str(self.files[index]) + '.txt'
        gt = np.loadtxt(file_gt)/580.0
        
        return data.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        return len(self.files)
        

cuda = torch.device('cuda:0')
learning_rate = 0.00005
momentum = 0.99
weight_decay = 0.0001
batch_size = 4

train_dataset = ReconDataset(True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataset = ReconDataset(False)
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
       
        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))
        
        writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
 
    
    torch.save(model.state_dict(), "/home/fenqiang/baobao/model.pkl")

