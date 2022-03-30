import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from numpy.testing import assert_almost_equal

class L_ent(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(L_ent, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float().cuda() + 0.5)

    def forward(self, y):
        b, _, h, w = y.shape
        y = y.reshape(b, 1, -1)
        c = self.centers.reshape(1, -1, 1).repeat(b, 1, 1)
        x = y - c
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))

        hist = torch.sum(x, 2)
        p = hist / (h * w) + 1e-6
        d = torch.sum((-p * torch.log(p)))
        return 1/d

class L_int(nn.Module):

    def __init__(self):
        super(L_int, self).__init__()

    def forward(self, x, mean_val, labels):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        d = torch.mean(torch.pow(x- labels,2))

        return d
        
class L_smo(nn.Module):
    def __init__(self):
        super(L_smo,self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size
