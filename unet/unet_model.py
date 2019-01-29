# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc1 = outconv(64, n_classes)
        
#        self.bn1 = nn.BatchNorm2d(32)
#        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
#        self.outc2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
#        self.bn2 = nn.BatchNorm2d(16)
#        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
#        self.outc3 = nn.ConvTranspose2d(16, 8, 2, stride=2)
#        self.bn3 = nn.BatchNorm2d(8)
#        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
#        self.outc4 = nn.ConvTranspose2d(8, 1, 2, stride=2)
#        self.last = nn.Upsample(size=tuple([9728,9728]), mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc1(x)
#        x = self.relu1(self.bn1(self.outc1(x)))
#        x = self.relu2(self.bn2(self.outc2(x)))
#        x = self.relu3(self.bn3(self.outc3(x)))
#        x = self.outc4(x)
#        x = self.last(x)
        
        return x
