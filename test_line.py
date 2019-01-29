#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:08:59 2019

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
        
        return data.astype(np.float32), gt.astype(np.float32), index

    def __len__(self):
        return len(self.files)
    
        
batch_size = 1
cuda = torch.device('cuda:0')

val_dataset = ReconDataset(False)
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
   
    np.savetxt('prediction/' + str(index[0].numpy()) + '_gt.txt', gt.cpu().numpy())
    np.savetxt('prediction/' + str(index[0].numpy()) + '_pred.txt', pred.detach().cpu().numpy())
