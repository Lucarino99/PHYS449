import numpy as np
import torch


import getopt, sys


short_options = "h"
long_options = ["help", "res-path=", "x-field=",
                "y-field=", "lb=", "ub=", "n-tests="]

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

#Info from the terminal
for current_argument, current_value in arguments:
    if current_argument in ("-h", "--help"):
        print ("Displaying help")
        jsonFile = open("./" + current_value)
    elif current_argument in ("", "--x-field"):
        x_field = current_value
    elif current_argument in ("", "--y-field"):
        y_field = current_value
    elif current_argument in ("", "--lb"):
        lb = float(current_value)
    elif current_argument in ("", "--ub"):
        ub = float(current_value)
    elif current_argument in("","--n-tests"):
        n_tests = int(current_value)
    elif current_argument in("","--res-path"):
        res_path = current_value



from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim

u = lambda x, y: eval(x_field)
v = lambda x, y: eval(y_field)

#Asymptotes of the field
asymptotes = [[0,0]]


randpoint = []

#Upper and lower bound
for i in range(n_tests):
    r = np.random.randn(1,2)

    while ((r[0][0] < lb) or (r[0][1] < lb)) or ((r[0][0] > ub) or (r[0][1] > ub)):
        r = np.random.randn(1,2)
    randpoint.append((r[0]).tolist())

#Parameters
num_epochs = 5000
learning_rate = 0.0001
displaystep = 75 #Keep
step_weight = 40 #Keep
divisions = 25

#The Network
Net = nn.Sequential(nn.Linear(2,500),
                    nn.Tanh(),
                     nn.Linear(500,50),
                    nn.Tanh(),
                     nn.Linear(50,2))

#Training data
x_train = []
y_train = []
for i in np.linspace(-1,1,divisions):
    x_train.append(i)
    y_train.append(i)

data_points = []

#Formatting
for i in range(len(x_train)):
    for j in range(len(y_train)):
        if x_train[i] == 0 and y_train[j] == 0:
            pass
        else:
            data_points.append([x_train[i],y_train[j]])

training_points = torch.tensor(data_points)

labels = []

#Results to get loss function from
for i in range(len(data_points)):
    labels.append([u(data_points[i][0],data_points[i][1]),v(data_points[i][0],data_points[i][1])])

labels = torch.tensor(labels)

criterion = nn.MSELoss()

optimizer = optim.SGD(Net.parameters(),learning_rate)

lowest_loss = 1

best_set_output = []
best_set_labels = []

#Get ouput
for e in range(num_epochs):

    running_loss = 0
    correct = 0

    #training_points -> NN -> Output
    output = Net(training_points.float())

    labels = labels.float()
    for i in range(len(labels)):
        if (output[i][0] == labels[i][0]) and (output[i][1] == labels[i][1]):
            correct += 1

    #Compare output(From NN) to true value
    loss = criterion(output,labels)
    loss.backward()
    optimizer.step()
    running_loss = running_loss + loss.item()

    if running_loss < lowest_loss:
        lowest_loss = running_loss
        best_set_output = output
        best_set_labels = labels


vector_field = best_set_output.tolist()
x_y = training_points.tolist()


fig, ax = plt.subplots()

field_dist = np.linspace(-1,1,10)
for i in range(10):
    for j in range(10):
        x = field_dist[i]
        y = field_dist[j]
        ax.quiver(x,y,u(x,y),v(x,y))


for i in range(n_tests):
    display = [randpoint[i]]
    display_x = [randpoint[i][0]]
    display_y = [randpoint[i][1]]
    ax.scatter(randpoint[i][0],randpoint[i][1])
    for i in range(displaystep):
        point = torch.tensor(display[i])

        gradient = (Net(point.float())).tolist()

        gradient[0] = gradient[0]/step_weight
        gradient[1] = gradient[1]/step_weight

        newpoint_0 = display[i][0] + gradient[0]
        newpoint_1 = display[i][1] + gradient[1]
        newpoint = [newpoint_0,newpoint_1]
        display_x.append(newpoint_0)
        display_y.append(newpoint_1)
        display.append(newpoint)

    ax.scatter(display_x,display_y,s=5)


plt.savefig(res_path)
plt.show()
