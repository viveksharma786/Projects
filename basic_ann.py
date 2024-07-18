

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import train_test_split

from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


input_path = "C:/Users/dh206356/Downloads/diabetes.csv"
print(input_path)
data = pd.read_csv(input_path)
data.head(2)

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values


Y_out =[] 
 
for z in Y:
    if z=="positive" :
        Y_out.append(1)
    else:
        Y_out.append(0)
        
        
Y_out = np.array(Y_out,dtype='float')

sc = StandardScaler()
X1 = sc.fit_transform(X)

X = torch.tensor(X1)
Y = torch.tensor(Y_out).unsqueeze(1)

print(Y.shape)

class Dataset(Dataset):
    
    def __init__(self,x,y):
        self.x =x
        self.y = y
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
dataset = Dataset(X,Y)

train_loader = torch.utils.data.DataLoader(dataset= dataset,
                            batch_size =100,
                            shuffle=True)

train_loader


print(len(train_loader))

for x,y in train_loader:
    print(x.shape)
    print(y.shape)
    

#7 input features so input layer will have 7 neurons

#build neural network

class Model(nn.Module):
    
    def __init__(self,input_feature, output_feature):
        super(Model,self).__init__()
        self.fc1 =nn.Linear(input_feature,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32,16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16,output_feature)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out=self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out
    

net = Model(7,1)

criterion =torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)

optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

# Define a learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

#number of epochs or batch size iteration

epochs = 200

#input --> forward pass --> loss calculation --> gardient calculation using backward pass --> weight updates

losses = []
lrs = []

for epoch in range(epochs):
    epoch_loss = 0
    for x, y in train_loader:
        x = x.float()
        y = y.float()
        #forward pass
        output = net(x)
        #loss calculation
        loss =criterion(output, y)
        #make existing gradient cleaned up
        optimizer.zero_grad()
        #grdient calculation
        loss.backward()
        #update weights
        optimizer.step()
        # Step the scheduler
        scheduler.step()
        epoch_loss += loss.item()
        
    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    #accuracy calculation
    final_output =(output > 0.5).float() # gives output [0,1,1,0...]
    accuracy = (final_output == y).float().mean()
    #print statistics
    print("epoch {}/{}, Loss: {:.3f},,Learning Rate : {}, Accuracy: {:.3f}".format(epoch+1, epochs, loss,scheduler.get_last_lr()[0], accuracy))
        

plt.plot(lrs, losses)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xscale('log')
plt.title('Loss vs. Learning Rate')
plt.show()
        



        








        
            



            
                
    