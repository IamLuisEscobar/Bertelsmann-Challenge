import torch

#Defining sigmoid activation function
def activation(x):
  return 1/(1+torch.exp(-x))
  
#Simple neural network
torch.manual_seed(7) #Set the random seed
features=torch.randn((1,5)) #Set five random values as inputs
weights=torch.randn_like(features) #Set random weights
bias=torch.randn((1,1)) #Set random bias
out=activation(torch.sum(features*weights)+bias) #Prediction of the neural network

#Same neural network but now we use torch.mm command which is faster than * operation 
weights=weights.view(5,1) #In order to use torch.mm we need to resize the weights
pred=activation(torch.mm(features,weights)+bias) #the value of pred is the same as out


#Neural network of two layers
torch.manual_seed(7)
#Features are 3 random variables
features=torch.randn((1,3))

#Define the size of each layer in the network
n_input=features.shape[1]
n_hidden=2
n_output=1

#Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
#Weights for hidden layer to output layer
W2 = torch.randn(n_hidden,n_output)

#Bias terms for hidden and outputs layers
B1 = torch.randn((1,n_hidden))
B2 = torch.randn((1,n_output)) 

first_layer_out=activation(torch.mm(features,W1)+B1)
second_layer_out=activation(torch.mm(first_layer,W2)+B2)
print(second_layer)
