
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dataset=pd.read_csv('/content/housing.csv')
dataset=dataset.dropna()
print("first ten records of dataset ")
dataset.head(10)

dataset1=dataset.loc[0:18,:]
#dataset1
graph1= dataset1.plot.line(subplots=True)
print("sub-plots of first eighteen samples from the dataset in a single figure.")
type(graph1)

graph2= dataset.plot.line(subplots=True)
print("sub-plots of all samples from the dataset in a single figure.")
type(graph2)

Y = dataset['median_house_value']
X = dataset.loc[:,'longitude':'median_income']
# print(Y)
# print(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()
x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

import torch
from torch.nn import Conv1d
from torch.nn import MaxPool1d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset

class cnnRegressor(torch.nn.Module):
  def __init__(self,batch_size, inputs, outputs):
    # initilize the super class & store the parameters
    super(cnnRegressor,self).__init__()
    self.batch_size= batch_size
    self.inputs= inputs
    self_outputs= outputs

    # define input  layers (input channels, output channels, kernel size)
    self.input_layer = Conv1d(inputs, batch_size,1)
    #(kernel size)
    self.max_pooling_layer= MaxPool1d(1)
    self.conv_layer= Conv1d(batch_size, 128,1)
    self.flatten_layer=Flatten()
    #(inputs, outputs)
    self.linear_layer= Linear(128,64)
    self.output_layer= Linear(64, outputs)

  def feed(self, input):
    #reshape the entry so it can fed to input layer
    input = input.reshape(self.batch_size, self.inputs,1)
    output=relu(self.input_layer(input))
    output=self.max_pooling_layer(output)
    output=relu(self.conv_layer(output))
    output=self.flatten_layer(output)
    output=self.linear_layer(output)
    output= self.output_layer(output)
    return output

#import SGD(stochastic gradient descent) package from pytroch for our optimizer
from torch.optim import SGD
#import mean aboslute error for our measure
from torch.nn import L1Loss

#import r^2 score package for score measure
!pip install pytorch-ignite
from ignite.contrib.metrics.regression.r2_score import R2Score

batch_size= 64
model= cnnRegressor(batch_size,X.shape[1],1)
#set the model to use the GPU for processing
model.cuda()

def model_loss(model, dataset, train=False, optimizer= None):
  #cycle through batches and get avg L1loss
  performance=L1Loss()
  score_metric=R2Score()

  avg_loss=0
  avg_score=0
  count=0
  for input, output in iter(dataset):
    # get the model predictions for training dataset
    predictions=model.feed(input)
    #get the model loss
    loss= performance(predictions, output)
    #get the model r2 score
    score_metric.update([predictions,output])
    score= score_metric.compute()

    if(train):
      #clear any errors so that they dont commulate
      optimizer.zero_grad()
      #compute gradiennts for our optimizer
      loss.backward()
      # use the optimizer to update the model parameters based on gradients
      optimizer.step()

    avg_loss +=loss.item()
    avg_score +=score
    count +=1
  return avg_loss / count, avg_score/count

epochs=10
optimizer= SGD(model.parameters(), lr=1e-5)


inputs=torch.from_numpy(x_train_np).cuda().float()
outputs=torch.from_numpy(y_train_np.reshape(y_train_np.shape[0],1)).cuda().float()

tensor=TensorDataset(inputs,outputs)
loader= DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

for epoch in range(epochs):
  avg_loss, avg_r2_score,= model_loss(model,loader,train=True, optimizer= optimizer)
  print("Epoch" + str(epoch +1)+ ":\n\tloss=" + str(avg_loss)+ "\n\tr^2 score=" + str(avg_r2_score))

inputs=torch.from_numpy(x_test_np).cuda().float()
outputs=torch.from_numpy(y_test_np.reshape(y_test_np.shape[0],1)).cuda().float()

tensor=TensorDataset(inputs,outputs)
loader= DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

avg_loss, avg_r2_score = model_loss(model, loader)
print("the model l1 loss is :" + str(avg_loss))
print("the model r2 score is :" + str(avg_r2_score))
