from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

from Adam import adam
from LBFGS import lbfgs
	
# path is a list of two lists, 1st containing all style paths and the second all content
# max_size is a list of tuples of dim for each content image
def load_img(combinations, max_size, preserve_color):
	'''
		path: list(style_path: list(str), content_path: list(str))
		max_size: tuple([w, h])
		returns two list of tensors
	'''
	print(max_size)
	def calc_dim(content, styles):
		style_dim = [style.size for style in styles]
		content_dim = content.size
		return tuple(min(dim) for dim in zip(content_dim, *style_dim))
		
	def transform(imgs):
		new = []
		for img in imgs:
			if preserve_color:
				img = cv.cvtColor(np.array(img, dtype=np.float32), cv.COLOR_RGB2YCrCb)		
			new.append(img_trans(img).unsqueeze(0).to(device, dtype=torch.float32))
		return torch.cat(new)
		
	def reshape(imgs, shape):
		new = []
		for img in imgs:
			new.append(img.resize(shape))
			#yield img.resize(shape)
		return new
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	styles = {}
	contents = {}
	for i, content in enumerate(combinations):
		contents[i] = Image.open(content).convert('RGB')
		styles[i] = []
		for style in combinations[content]:
			styles[i].append(Image.open(style).convert('RGB'))
	
	if len(max_size) == 0:
		max_size = [calc_dim(contents[index], styles[index]) for index in contents]
	
	if len(max_size) == 1:
		max_size = max_size*len(contents)
		# same dimension for each set
		
	print(max_size)
	
	img_trans = transforms.Compose([
					transforms.ToTensor(),
					# values used in ImageNet dataset
					transforms.Normalize((0.485, 0.456, 0.406), 
										(0.229, 0.224, 0.225))
				])
		
	for index in contents:
		contents[index] = transform(reshape([contents[index]], max_size[index]))
		styles[index] = transform(reshape(styles[index], max_size[index]))


	print(styles[0].shape)
	print(contents[0].shape)

	#requires_grad = False by default
	return contents, styles

def tensor2img(tensor, preserve_color):
	'''
		tensor: torch.tensor of dimension 1*C*W*H
		returns ndarray W*H*c
	'''
	img = tensor.to(device='cpu', dtype=torch.float32).clone().squeeze(0).detach()
	img = img.numpy()
	img = img.transpose(1, 2, 0)	
	img = img * np.array((0.229, 0.224, 0.225), dtype=np.float32) + np.array((0.485, 0.456, 0.406), dtype=np.float32)
	
	if preserve_color:
		img = cv.cvtColor(img, cv.COLOR_YCrCb2RGB)
	return img.clip(0, 1)

def VGG19Style():
	'''
		returns torch.Sequential of VGG19 pretrained, ReLU replaced with avg pooling
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vgg = models.vgg19(pretrained=True).features.to(device)

	layers = []
	for layer in vgg:
		if type(layer) != torch.nn.modules.pooling.MaxPool2d:
			if type(layer) == torch.nn.modules.activation.ReLU:
				layers.append(nn.ReLU(inplace=False))
			else:
				layers.append(layer)
		else:
			layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

	Sequential = nn.Sequential(*layers)
	del layers, layer, vgg

	for param in Sequential.parameters():
		param.requires_grad = False

	return Sequential.eval()
	
def extract_features(imgs, model):

	'''
		img: torch.tensor
		model: torch.Sequential
		returns a dict of needed tensors in a list from model
	'''
	
	conv_blocks = {
		2: 'conv1_1',
		7: 'conv2_1',
		12: 'conv3_1',
		21: 'conv4_1',
		23: 'conv4_2',
		30: 'conv5_1'
	}
	
	features = {}
	
	for i, layer in enumerate(model):
		imgs = layer(imgs)

		if i in conv_blocks:
			features[conv_blocks[i]] = imgs

	return features

def gram_matrix(tensor):
	n, c, h, w = tensor.shape
	tensor = tensor.view(n, c, h*w)
	tensor_trans = torch.transpose(tensor, 1, 2)
	return torch.matmul(tensor, tensor_trans)

def train(combinations, results, result_dir, beta, epochs, max_size, optim_fn, verbose, preserve_color):
	
	'''
		combinations: dict(content_path: style_path: list)
		results: list
		result_dir: str
		beta: float
		epochs: float
		max_size: [w: int, h: int]
		optim_fn: str
	'''
	print(max_size)
	contents, styles = load_img(combinations, max_size, preserve_color)

	model = VGG19Style()
	
	style_weights = {
	    'conv1_1': 1,
	    'conv2_1': 0.75,
	    'conv3_1': 0.2,
	    'conv4_1': 0.2,
	    'conv5_1': 0.2
	}
	
	epochs = int(float(epochs))
	alpha = 1
	beta = float(beta)
		
	for index in contents:
		print(f'Working on {results[index][:-4]}')
		losses = {
			'style': ['tab:orange', []],
			'content': ['tab:green', []],
			'total': ['tab:red', []]
		}

		style_features = extract_features(styles[index], model)
		content_features = extract_features(contents[index], model)
		print(style_features['conv2_1'].shape)
		print(content_features['conv2_1'].shape)
		style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
		print(style_grams['conv2_1'].shape)
			
		features = (style_grams, content_features)
		
		if optim_fn == 'adam':
			target = contents[index].clone().requires_grad_(True)
			show_every = 50
			params = (model, alpha, beta, style_weights, show_every, epochs, verbose)
			target = adam(features, target, params, losses)
		elif optim_fn == 'lbfgs':
			show_every = 1
			target = torch.rand(size=content.shape, dtype=content.dtype, device=content.device, requires_grad=True)
			params = (model, alpha, beta, style_weights, show_every, epochs, verbose)
			target = lbfgs(features, target, params, losses)
		else:
			print('Current optimizer not supported')
			sys.exit(-1)

		fig, ax = plt.subplots(len(losses), sharex=True)
		fig.suptitle(f'Training losses beta:{beta} epochs{epochs}')

		for plot, loss in enumerate(losses):
			ax[plot].plot(range(1, epochs+1, show_every), losses[loss][1], losses[loss][0], label=loss)
			ax[plot].legend()

		print(result_dir+'Loss'+results[index])
		fig.savefig(result_dir+'Loss'+results[index])
		plt.imsave(results[index], tensor2img(target, preserve_color))
		print('*'*20)
