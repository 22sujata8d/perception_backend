import sys
sys.path.append("/home/sujata/Backend/model")

import os	
import torch
import model
import numpy as np

from torchvision import transforms
from PIL import Image, ImageOps
from matplotlib import cm

from model.classifier import Classifier

def load_checkpoint(filepath):
	""" Returns the loaded model from the saved .pt or .pth file..

		Args:
		----
		filepath: path where the saved model file is present (of extension .pt or .pth)
	"""

	# loading the saved checkpoint from "checkpoint.pth"
	checkpoint = torch.load(filepath)
	# loading the saved model
	model = checkpoint['model']
	# loading saved model parameters
	model.load_state_dict(checkpoint['state_dict'])

	# setting require grad false so that model can be used for prediction 
	# rather than training.
	for parameter in model.parameters():
		parameter.require_grad = False

	# set the model for evaluation mode
	model.eval()

	return model 

def get_probabilities(img):
	""" Returns the probabilities for different output labels 0 to 9.

		Args:
		-----
		imag_arr: nd-array with size of (28, 28)

		Return:
		------
		ps: torch.Tensor of size (1, 10) contains probabilities for digits 0 to 9. 
	"""
	
	#print(type(img_arr))

	# convert the image from nd-array to PIL.Image format
	#img = Image.fromarray(np.uint8(cm.gist_earth(img_arr)*255))

	#img = img.convert('L')

	img.save("your_file.jpeg")

	#img = img_arr
	print(type(img))

	# load the saved model
	model = load_checkpoint('/home/sujata/multi_layer_perceptron/Backend/model/checkpoint.pth')
	
	# convert the image to greyscale 
	gray_image = ImageOps.grayscale(img)

	# define a transform to normalize the data
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
	
	# convert the image to tensor by using transform
	transform_img = transform(gray_image)
	
	print(type(transform_img))
	print(transform_img.shape)

	# get the probabilities from the saved model
	ps = torch.exp(model(transform_img))

	# print the probabilities
	print(ps)
	
	return ps

	
