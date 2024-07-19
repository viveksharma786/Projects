# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:34:15 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.datasets

x,y = sklearn.datasets.make_moons(n_samples=200,noise=0.20)

plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Spectral)

x= torch.tensor(x).float()

y = torch.tensor(y)


class FeedForward(torch.nn.Module):
    
    def __init__(self, x,hidden_neurons,y):
        super(FeedForward,self).__init__()
        self.linear = nn.Linear(x,hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, y)
        self.Relu = nn.ReLU()
        
    def forward(self,x):
        out = self.linear(x)
        out = self.Relu(out)
        out = self.output_layer(out)
        return out
    
net = FeedForward(2,50,2)

loss_function = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.SGD(net.parameters(),lr=0.001,momentum = 0.01)

epochs =10000


for epoch in range(epochs):
    out = net(x)
    loss= loss_function(out,y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    if epoch % 1000 == 0:
        max_value, prediction = torch.max(out,1)
        predicted_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],s=40, c=predicted_y,lw=0)
        accuracy = (predicted_y==target_y).sum()/target_y.size
    print("epochs :{} , loss = {:.3f}, accuracy = {:.3f}".format(epoch,loss, accuracy))
        
       
#check two classes values

output = out.data.numpy()
        
        
        
    
    
        

