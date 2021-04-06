import torch
import torch.nn as nn
import torch.optim as optim

import main

def lbfgs(features, target, params, losses):

	style_grams, content_features = features
	model, alpha, beta, style_weights, show_every, epochs, verbose = params
	
	for key in losses:
		losses[key][1] = [0]*epochs
	
	optimizer = optim.LBFGS([target])

	for epoch in range(epochs):		
		
		def closure():
			closure.counter += 1
			target_features = main.extract_features(target, model)
			target_grams = {layer: main.gram_matrix(target_features[layer]) for layer in target_features}
							
			### content loss ###
			content_loss = 0.5*torch.sum((content_features['conv4_2'] - target_features['conv4_2'])**2)
			
			### stye loss ###			
			style_loss = 0
			for layer in style_weights:
				style_loss += style_weights[layer]*torch.sum((style_grams[layer] - target_grams[layer])**2)/(2*target_grams[layer].shape.numel())**2
			
			### total loss ###
			total_loss = alpha*content_loss + beta*style_loss			
			
			if not closure.counter and (epoch%show_every == 0):
			
				if verbose:
					print(f'Epoch {epoch + 1} style loss is {style_loss}')
					print(f'Epoch {epoch + 1} content loss is {content_loss}')
	
				losses['style'][1][epoch] = style_loss.item()
				losses['content'][1][epoch] = content_loss.item()
				losses['total'][1][epoch] = total_loss.item()
			
			optimizer.zero_grad()
			total_loss.backward()
			return total_loss
		closure.counter = -1
		optimizer.step(closure=closure)
			
	return target
