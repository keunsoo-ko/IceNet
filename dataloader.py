import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from skimage.color import rgb2ycbcr
from skimage.segmentation import slic

random.seed(1143)

import numpy as np
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
	# Padding
	A = np.pad(A, padding, mode='reflect')

	# Window view of A
	output_shape = ((A.shape[0] - kernel_size)//stride + 1,
			(A.shape[1] - kernel_size)//stride + 1)
	kernel_size = (kernel_size, kernel_size)
	A_w = as_strided(A, shape = output_shape + kernel_size, 
		strides = (stride*A.strides[0],
			stride*A.strides[1]) + A.strides)
	A_w = A_w.reshape(-1, *kernel_size)

	# Return the result of pooling
	if pool_mode == 'max':
		return A_w.max(axis=(1,2)).reshape(output_shape)
	elif pool_mode == 'median':
		return A_w.median(axis=(1,2)).reshape(output_shape)
	elif pool_mode == 'avg':
		return A_w.mean(axis=(1,2)).reshape(output_shape)


def train_list(lowlight_images_path):
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	random.shuffle(train_list)
	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = train_list(lowlight_images_path) 
		self.size = 256
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):

		data_lowlight_path = self.train_list[index]
		data_lowlight = Image.open(data_lowlight_path)
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight = np.asarray(data_lowlight)
		ycbcr = rgb2ycbcr(data_lowlight)
		y = ycbcr[..., 0]


		data_lowlight = data_lowlight / 255.
		y = y / 255.
 
		data_lowlight = torch.from_numpy(data_lowlight).float()
		y = torch.from_numpy(y).float()
		m = np.float32(random.randint(20, 80) / 100.)

		scribble = np.float32(np.zeros([self.size, self.size]))

		positive = random.randint(0, 5)
		negative = random.randint(0, 5)
		for _ in range(positive):
			x_ = np.random.randint(0, self.size)
			y_ = np.random.randint(0, self.size)
			cv2.circle(scribble, (x_, y_), 20, 1, -1)
		for _ in range(negative):
			x_ = np.random.randint(0, self.size)
			y_ = np.random.randint(0, self.size)
			cv2.circle(scribble, (x_, y_), 20, -3, -1)

		t = np.clip(y + scribble * 15. / 255., 0., 1.)
		Ga = t ** 0.2
		Gb = 1-(1-t) ** 0.2
		
		G = (m * Ga + (1-m) * Gb)

		labels = pool2d(G, 15, 1, 15//2, pool_mode='max')

		return data_lowlight.permute(2,0,1), y[None], m, labels[None], scribble[None]

	def __len__(self):
		return len(self.train_list)

