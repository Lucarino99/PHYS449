import torch
import torchvision

import sys
import json

if (len(sys.argv) != 2):
    print("INVALID PARAMETERS CODE SHOULD BE EXECUTED IN THE FOLLOWING FORMAT:")
    print("python main.py param/param_file_name.json")
else:

    print(sys.argv[1])

    jsonFile = open("./" + sys.argv[1])

    paramData = json.load(jsonFile)

    print(paramData)

    n_epochs = 1000
    l_r = 0.0001

from data_gen import Data

import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

Net = nn.Sequential(nn.Linear(196, 80),
                      nn.ReLU(),
                      nn.Linear(80, 40),
                      nn.ReLU(),
                      nn.Linear(40,20),
                      nn.ReLU(),
                      nn.Linear(20, 5),
                      nn.LogSoftmax(dim=1))
        #We have 196 input nodes, 50 hidden nodes and 5 output nodes


data = Data()

#Defines the loss
criterion = nn.NLLLoss()

#Optimizer
optimizer = optim.SGD(Net.parameters(),l_r)

print("Beginning Network Training....\n")

for e in range(int(paramData["num_epochs"])):
    running_loss = 0
    x_train = data.x_train
    output = Net((x_train).float())

    y = data.y_train
    y = torch.div(y, 2)
    y = (y.type(torch.LongTensor))

    correct = 0

    for j in range(len(y)):
        if(torch.argmax(output[j]) == y[j]):
            correct += 1



    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    running_loss = running_loss + loss.item()
    if (e % 10) == 0:
        print("Epoch : [",e,"/",int(paramData["num_epochs"]),"]")
        print("", "Training Loss: ", round((running_loss/10),10), " | Training Accuracy: ", round(((correct/len(y))*100),2), "%")

    #Testing Code
x_test = data.x_test
output = Net((x_test).float())

y_test = data.y_test
y_test = torch.div(y_test, 2)
y_test = (y_test.type(torch.LongTensor))

correct = 0

for j in range(len(y_test)):
    if (torch.argmax(output[j]) == y_test[j]):
            correct += 1

loss = criterion(output, y_test)

if (e % 10) == 0:
    print("Testing Loss: ", round(((loss.item() / 10)),10), " | Testing Accuracy: ", round(((correct / len(y_test)) * 100),2), "%\n")
