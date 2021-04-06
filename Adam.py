import torch
import torch.nn as nn
import torch.optim as optim

import main

def adam(features, target, params, losses):
	style_grams, content_features = features
	model, alpha, beta, style_weights, show_every, epochs, verbose = params
	
	optimizer = optim.Adam([target], lr=0.002)
	
	for epoch in range(epochs):
		target_features = main.extract_features(target, model)
		target_grams = {layer: main.gram_matrix(target_features[layer]) for layer in target_features}
		
		### content loss ###
		content_loss = 0.5*torch.sum((content_features['conv4_2'] - target_features['conv4_2'])**2)

		### style loss ###
		style_loss = 0
		for layer in style_weights:
			style_loss += style_weights[layer]*torch.sum((style_grams[layer] - target_grams[layer])**2)/(2*target_grams[layer].shape[1:].numel())**2
		
		### total loss ###	
		total_loss = alpha*content_loss + beta*style_loss
		
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		
		if epoch%show_every == 0:
			if verbose:
				print(f'Epoch {epoch + 1} style loss is {style_loss.item()}')
				print(f'Epoch {epoch + 1} content loss is {content_loss.item()}')

			losses['style'][1].append(style_loss.item())
			losses['content'][1].append(content_loss.item())
			losses['total'][1].append(total_loss.item())
		
		#target.data.clamp_(-1, 1)
	return target
