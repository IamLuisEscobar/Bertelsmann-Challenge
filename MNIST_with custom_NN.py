from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import helper
import torch

def activation(x):
  return 1/(1+torch.exp(-x))

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');


#Flatten the input images
inputs=images.view(images.shape[0], -1) #The values are (64,784)

#Parameters
W1=torch.rand(784,256)
B1=torch.rand(256)

W2=torch.rand(256,10)
B2=torch.rand(10)

out1=activation(torch.mm(inputs,W1)+B1)
print(out1.shape)

final_values=torch.mm(out1,W2)+B2
final_values.shape

torch.sum(final_values[0])

#My softmax solution
def exp_sum(vector):
  result=[]
  for values in vector:
    result.append(torch.exp(values))
  return sum(result)

def my_softmax(vector):
  softmax_vector=[]
  sum_value=exp_sum(vector)
  for values in vector:
    softmax_vector.append(torch.exp(values)/sum_value)
  return softmax_vector
  
#The fancy softmax solution
def softmax(x):
  return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)
