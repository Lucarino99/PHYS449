import numpy as np
import pandas as pd

df = pd.read_csv('./data/in', header=None)
json = str(pd.read_csv('./data/mods.json', error_bad_lines=None))

beg = 0
end = 0
L = []
for i in range(len(json)):
    if json[i] == ':':
        beg = i
        L.append(beg)
    if json[i:i+3] == "NaN":
        end = i
        L.append(end)

L = L[0:4]
l_r_range = L[0:2]
num_iter_range = L[2:4]
l_r = json[L[0]:L[1]]
num_iter = json[L[2]:L[3]]

l_r = l_r.replace(' ','')
l_r = float(l_r.replace(':',''))
num_iter = num_iter.replace(' ','')
num_iter = int(num_iter.replace(':',''))




#Isolating for l_r:


#First, I want to filter through the data and get arrays of respective numbers
XMatrix = []

def filter_through_df(df):
    new_append = []
    for i in range(len(df)):
        XMatrix.append(df[0][i])


filter_through_df(df)
#We now want to make this vector of strings an array of numbers
def filter_through_spaces(XMatrix):
    NewMatrix = []
    yVector = []
    for i in range(len(XMatrix)):
        x = 0
        L = [1]
        XMatrix_updated = []
        for j in range(len(XMatrix[i])):
            if XMatrix[i][j] == " ":
                L.append(float(XMatrix[i][x:j]))
                x = j + 1
            if j+1 == len(XMatrix[i]):
                yVector.append(float(XMatrix[i][x:j+1]))
        NewMatrix.append(L)

    return(NewMatrix,yVector)

XMatrix, yVector = filter_through_spaces(XMatrix)

#We now have our XMatrix and our y-vector. We'll now begin by having a
# sample hypthesis w(transpose) dotted with our x vector.

#Here, we're effectively allowing our initial omegavector values to be
# 1 each. And our starting X vector is the first row of the XMatrices.
w = [0 for i in range(len(XMatrix[0]))]

#Now, to change our w parameters:

for k in range(num_iter):
    for i in range(len(XMatrix)):
        for j in range(len(w)):
            gradCost = (1 / len(XMatrix)) * (np.dot(w, XMatrix[i]) - yVector[i]) * XMatrix[i][j]
            w[j] = w[j] - (l_r * gradCost)


f=open('./data/out','w+')
for i in range(len(w)):
    f.write(str(w[i]))
    f.write("%d\r\n" % (i + 1))

f.close()


