import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class IceNet(nn.Module):

	def __init__(self):
		super(IceNet, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.e_conv1 = nn.Conv2d(2,32,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(32,32,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(32,32,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(32,32,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(64,32,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(64,32,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(64,32,3,1,1,bias=True)

		self.fc1 = nn.Linear(1, 32)
		self.fc2 = nn.Linear(32, 32)


		
	def forward(self, y, maps, e, lowlight=None, is_train=False):
		b, _, h, w = y.shape
		x_ = torch.cat([y, maps], 1)

		# generate adaptive vector according to eta
		W = self.relu(self.fc1(e))
		W = self.fc2(W)

		# feature extractor
		x1 = self.relu(self.e_conv1(x_))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x_r = self.relu(self.e_conv7(torch.cat([x1,x6],1)))

		# AGEB
		x_r = F.conv2d(x_r.view(1, b * 32, h, w),
				W.view(b, 32, 1, 1), groups=b)
		x_r = torch.sigmoid(x_r).view(b, 1, h, w) * 10

		# gamma correction
		enhanced_Y = torch.pow(y,x_r)
		if is_train:
			return enhanced_Y, x_r
		else:
			# color restoration
			enhanced_image = torch.clip(enhanced_Y*(lowlight/y), 0, 1)
			return enhanced_image
