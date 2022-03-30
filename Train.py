import torch
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
	IceNet = model.IceNet().cuda()
	IceNet.apply(weights_init)
	if config.load_pretrain:
	    IceNet.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_E = Myloss.L_ent(256, 0., 255., 10)

	L_I = Myloss.L_int()
	L_S = Myloss.L_smo()


	optimizer = torch.optim.Adam(IceNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	IceNet.train()

	for epoch in range(config.num_epochs):
		for iteration, batch in enumerate(train_loader):
			img_lowlight = batch[0].cuda()
			Y = batch[1].cuda()
			m = batch[2][:, None].cuda()
			labels = batch[3].cuda()
			scribble = batch[4].cuda()

			enhanced_Y, A = IceNet(Y, scribble, m, is_train=True)

			enhanced_image = enhanced_Y*(img_lowlight/Y)

			loss_s = 20*L_S(A)
			loss_e = 10*L_E(enhanced_Y * 255.)

			loss_i = torch.mean(L_I(enhanced_Y, m, labels))
			# best_loss
			loss =  loss_s + loss_e + loss_i
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(IceNet.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Epoch{}, Loss at iteration{} : {}".format(epoch, iteration+1, loss.item()))
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(IceNet.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
