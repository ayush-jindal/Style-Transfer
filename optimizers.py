import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

import main

def optimize(model, features, target, optim_fn, closure, epochs, params, losses):
	for key in losses:
		losses[key][1] = []
	print((epochs//params[-1]))
	
	if optim_fn == 'adam':
		optimizer = optim.Adam([target], lr=0.002)
	elif optim_fn == 'lbfgs':
		optimizer = optim.LBFGS([target])
	elif optim_fn == 'sgd':
		optimizer = optim.SGD([target], lr=1e-6, momentum=0.9)
		
	def closure_fn(features, params, target):
		target_features = main.extract_features(target, model)
		target_grams = {layer: main.gram_matrix(target_features[layer]) for layer in target_features}
		return loss(features, params, target_features, target_grams)					
	
	#loss.counter = -1
	def loss(features, params, target_features, target_grams):
		loss.counter += 1
		
		style_grams, content_features = features
		alpha, beta, style_weights, verbose, show_every = params
		### content loss ###
		content_loss = 0.5*torch.sum((content_features['conv4_2'] - target_features['conv4_2'])**2)
		
		### stye loss ###			
		style_loss = 0
		for layer in style_weights:
			style_loss += style_weights[layer]*torch.sum((style_grams[layer] - target_grams[layer])**2)/(2*target_grams[layer].shape[1:].numel())**2
		
		### total loss ###
		total_loss = alpha*content_loss + beta*style_loss			
		
		if not loss.counter and (epoch%show_every == 0):
		
			if verbose:
				print(f'Epoch {epoch + 1} style loss is {style_loss}')
				print(f'Epoch {epoch + 1} content loss is {content_loss}')

			losses['style'][1].append(style_loss.item())
			losses['content'][1].append(content_loss.item())
			losses['total'][1].append(total_loss.item())
		
		optimizer.zero_grad()
		total_loss.backward()
		return total_loss	

	for epoch in range(epochs):
		loss.counter = -1
		if not closure:
			target_features = main.extract_features(target, model)
			target_grams = {layer: main.gram_matrix(target_features[layer]) for layer in target_features}
			loss(features, params, target_features, target_grams)
			optimizer.step()
		else:
			optimizer.step(closure=lambda: closure_fn(features, params, target))
			
	return target
