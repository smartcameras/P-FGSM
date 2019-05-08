"""
Scene privacy protection
C.Y. Li, A.S. Shamsabadi, R. Sanchez-Matilla, R. Mazzon, A. Cavallaro
Proc. of IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), Brighton, UK, May 12-17, 2019
If you find this code useful in your research, please consider citing
Please check the license of this code in License.txt
"""

import csv
import os
from os.path import join,isfile
import copy

import urllib

import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F

cuda = torch.cuda.is_available()

def loadModel(args):
	models_directory = './models'
	if not os.path.exists(models_directory):
		os.makedirs(models_directory)

	# Check if the model exists, otherwise download it
	model_file = '{}/{}_places365.pth.tar'.format(models_directory, args.model)
	if not os.access(model_file, os.W_OK):
	    weight_url = 'http://places2.csail.mit.edu/models_places365/' + args.model + '_places365.pth.tar'
	    print(weight_url)
	    os.system('wget ' + weight_url)
	    os.system('mv *.pth.tar ./models')
	    print('Model downloaded!')

	model = models.__dict__[args.model](num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	model.eval()
	if cuda:
		model.cuda()
	return model

def loadImagePaths(args):
	return [f for f in os.listdir(args.path) if isfile(join(args.path,f))]

def createLogFiles(args):
	f_name = './log.txt'
	f = open(f_name,"w")
	return f, f_name

def createDirectories(args):
	adv_path = '{}/adversarials'.format(args.path)
	if not os.path.exists(adv_path):
		os.makedirs(adv_path)

	return adv_path


def readImage(args, imagePaths, index):

	img_name = imagePaths[index]
	org_image = cv2.imread(join(args.path,img_name), 1)
	org_image = cv2.resize(org_image, (224, 224), interpolation=cv2.INTER_LINEAR)

	# Image boundaries
	lb, ub = calculate_bounds(org_image, args.eps)

	return img_name, org_image, lb, ub

def calculateSetTargetClasses(org_image, model, args):

	# Process image
	image = preprocess_image(org_image)

	# Forward pass
	logit = model.forward(image)
	h_x = F.softmax(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)
	probs = np.array(probs.cpu())
	idx = np.array(idx.cpu())

	# Calculate set of target classes
	cumSum = np.cumsum(probs)
	set_target_classes = idx[cumSum > args.sigma]
	set_target_classes = idx[1:]

	return (idx[0], probs[0], set_target_classes)

def addNoise(_x, _noise, lb, ub):

	x = _x.clone()
	noise = _noise.clone()

	# De-standarise image
	x = destandarise(x)
	
	# Add noise
	x -= noise

	# Clip
	x = torch.min(torch.max(x, lb), ub)

	# Standarise image
	x = standarise(x)

	return x

def calculate_bounds(img, epsilon):
   
	img = np.float32(img)
	img = np.ascontiguousarray(img[..., ::-1])
	img = img.transpose(2, 0, 1)  # Convert array to D,W,H
	
	# Normalize the channels
	for c, _ in enumerate(img):
		img[c] /= 255

	img = torch.from_numpy(img).float()
	img.unsqueeze_(0)

	lb = Variable(img.clone(), requires_grad=False)
	ub = Variable(img.clone(), requires_grad=False)
 
	# Compute the bounds
	lb -= epsilon
	ub += epsilon

	# Clip between [0,1]
	lb = torch.clamp(lb, 0., 1.)
	ub = torch.clamp(ub, 0., 1.)
	if cuda:
		lb = lb.cuda()
		ub = ub.cuda()

	#cv2.imwrite('./lb.png', recreate_image(lb))
	#cv2.imwrite('./ub.png', recreate_image(ub))
	return lb, ub

def preprocess_image(cv2im, resize_im=True):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	
	# Resize image
	if resize_im:
		cv2im = cv2.resize(cv2im, (224, 224), interpolation=cv2.INTER_LINEAR)
	im_as_arr = np.float32(cv2im)
	im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
	im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
	# Normalize the channels
	for channel, _ in enumerate(im_as_arr):
		im_as_arr[channel] /= 255
		im_as_arr[channel] -= mean[channel]
		im_as_arr[channel] /= std[channel]
	# Convert to float tensor
	im_as_ten = torch.from_numpy(im_as_arr).float()
	# Add one more channel to the beginning. Tensor shape = 1,3,224,224
	im_as_ten.unsqueeze_(0)
	# Convert to Pytorch variable
	if cuda:
		im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
	else:
		im_as_var = Variable(im_as_ten, requires_grad=True)

	return im_as_var
	

def recreate_image(im_as_var):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
	for c in range(3):
		recreated_im[c] *= std[c]
		recreated_im[c] += mean[c]
	recreated_im[recreated_im > 1] = 1
	recreated_im[recreated_im < 0] = 0
	recreated_im = np.round(recreated_im * 255)
	recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
	# Convert RBG to GBR
	recreated_im = recreated_im[..., ::-1]
	return recreated_im


def destandarise(img):
	
	img2 = img.clone()

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for c in range(3):
		img2[0][c] *= std[c]
		img2[0][c] += mean[c]

	return img2

def standarise(img):
	
	img2 = img.clone()

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	for c in range(3):
		img2[0][c] -= mean[c]
		img2[0][c] /= std[c]
		
	return img2