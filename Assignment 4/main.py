import math

from matplotlib import pyplot as plt

import torch.utils.data

#We want to load the training data

import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]

training_set = lines

learning_rate = 0.0001
num_epochs = 100



#We now want to turn this list of strings into a list of integers, -1 or 1.

new_set = []
for i in range(len(training_set)):
    L = []
    for j in range(len(training_set[0])):
        if training_set[i][j] == '-':
            L.append(-1)
        if training_set[i][j] == '+':
            L.append(1)
    new_set.append(L)

#Make coupling constants for arbitray row dimensions.
size = len(training_set[0])

couplingconst = (torch.randint(2,(size,))).tolist()

for i in range(len(couplingconst)):
    if couplingconst[i] == 0:
        couplingconst[i] = -1

def eval_energy(listoflists,couplingcs):
    energy_of_ith = []
    for i in range(len(listoflists[0])):
        add = 0
        if i != (len(listoflists[0]) - 1):
            for j in range(len(listoflists)):
                add += listoflists[j][i] * listoflists[j][i+1] * couplingcs[i]
        else:
            for j in range(len(listoflists)):
                add += listoflists[j][0] * listoflists[j][len(listoflists[0])-1] * couplingcs[i]
        energy_of_ith.append(add)

    return(energy_of_ith)

print(sum(eval_energy(new_set,[-1,1,1,1])))
#Positive Phase

loss = []
for n in range(num_epochs):

    PositivePhase = eval_energy(new_set,couplingconst)
#Now for Negative Phase

#We first need to generate variations of our real data set,

    num_iterations = 100

    X = new_set
    for i in range(num_iterations):
        Y = torch.randint(0,2,(len(new_set),len(new_set[0])))
        Y = Y.tolist()
        for j in range(len(Y)):
            for k in range(len(Y[j])):
                if Y[j][k] == 0:
                    Y[j][k] = -1

        EnergyX = sum(eval_energy(X,couplingconst))
        EnergyY = sum(eval_energy(Y,couplingconst))

        if EnergyY < EnergyX:
            X = Y
        else:
            prob = math.exp(EnergyX-EnergyY)
            prob = torch.tensor([prob])
            change = torch.bernoulli(torch.tensor(prob))
            change = change.tolist()

            if change[0] == 1.0:
                X = Y

    NegativePhase = eval_energy(X,couplingconst)

    #Update rule


    for i in range(len(PositivePhase)):
        UpdateValues = (PositivePhase[i]/len(PositivePhase)) - (NegativePhase[i]/len(NegativePhase))
        couplingconst[i] = couplingconst[i] + learning_rate * UpdateValues

    if (n % 10 == 0):
        print("Energy of the system is: ",sum(PositivePhase)," in epoch [",n,",",num_epochs,"]")
        print(couplingconst)
        print(" ")

    loss.append([-sum((PositivePhase)) + sum((NegativePhase)),n])


print("The final coupling constants are then: {(0,1): ",couplingconst[0],"(1,2): ",couplingconst[1],"(2,3): ",couplingconst[2],"(3,0): ",couplingconst[3],"}.")

plt.plot(loss)
plt.show()
