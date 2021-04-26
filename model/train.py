import numpy as np
import pandas as pd
import torch

import helper

from classifier import Classifier

from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

################ Import libraries ######################
# uncomment below lines if code runs on jupyter notebook

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')

#########################################################

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

############### Visualize downloaded mnist image ##########
# uncomment below lines if code runs on jupyter notebook

# image, label = next(iter(trainloader))
# helper.imshow(image[0,:]);

###########################################################

# Create the network, define the criterion and optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


######################
# Training the Model #
######################

epochs = 5

train_losses, test_losses = [], []

for e in range(epochs):
    
    running_loss = 0
    tot_train_loss = 0

    for images, labels in trainloader:
        # forward propogation 
        log_ps = model(images)
        # calculating loss using cross entropy
        loss = criterion(log_ps, labels)
        
        # backpropogation step
        optimizer.zero_grad()
        loss.backward()
        # optimising the values of weights
        optimizer.step()
        
        running_loss += loss.item()

    else:

        tot_test_loss = 0
        # Number of correct predictions on the test set
        test_correct = 0  
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                tot_test_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()

        # Get mean loss to enable comparison between train and test sets
        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))

################ Plot the graph for validation & training loss ####
# uncomment below lines if code runs on jupyter notebook

# Plotting the validation and training losses
# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Validation loss')
# plt.legend(frameon=False)

########### Test out the network #########################
# uncomment the following if code runs on jupyter notebook 

# dataiter = iter(testloader)
# images, labels = dataiter.next()
# img = images[1]

# Calculate the class probabilities (softmax) for img
# ps = torch.exp(model(img))

# Plot the image and probabilities
# helper.view_classify(img, ps, version='Digits')

###############################################################

# Save the model
checkpoint = {'model': Classifier(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
