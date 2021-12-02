import torch
import sys, getopt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")

n = 0
resultDir = ""
try:
    opts, args = getopt.getopt(sys.argv[1:], "vo:n:", ["--help"])
except getopt.GetoptError as err:
    print("Must be executed in form: python main.py -v verbose -o result_dir -n samples")
    sys.exit(2)
verbose = False
for o, a in opts:
    if o == "-v":
        verbose = True
    elif o == "-o":
        resultDir = a
    elif o == "-n":
        n = int(a)
    elif o == "--help":
        print("-v: Enables the creation of iterative reports on loss value")
        print("-o: Defines the directory where results are printed out")
        print("-n: Defines the number of samples that will be printed out")
        sys.exit(2)
    else:
        assert False, "unhandled option"

if not os.path.isdir(resultDir):
    os.makedirs(resultDir)

if n <= 0:
    print("N MUST BE GREATER THAN 0")
    sys.exit(2)


#Hyperparameters

batch_size = 128
learning_rate = 1e-3
num_epochs = 25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*6*6, zDim=256):
        super(VAE, self).__init__()

        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = x.view(-1, 32*6*6)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 6, 6)
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar



with open('data/even_mnist.csv') as f:
    lines = f.read().splitlines()

images = []
labels = []

for i in range(len(lines)):
    image = list(map(int, lines[i].split(' ')))
    label = image[-1]
    del(image[-1])
    M = []
    for j in range(14):
        row = image[14*j:14+14*j]
        M.append(row)
    images.append(M)
    labels.append(label)

images = torch.Tensor(images)
labels = torch.Tensor(labels)

trainingImgs,testingImgs = images.split(29192)
trainingLbls, testingLbls = labels.split(29192)


trainData = datasets.MNIST('data', train=True, transform=transforms.ToTensor())
testData = datasets.MNIST('data', train=False, transform=transforms.ToTensor())

trainData.data = trainingImgs
trainData.train_labels.data = trainingLbls

testData.data = testingImgs
testData.train_labels.data = testingLbls


train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testData,batch_size=1,shuffle=True)

net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


###Training
L = []
line = []
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)

        out, mu, logVar = net(imgs)

        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    L.append([loss,epoch])
    lin = 'Epoch : [' + str(epoch+1)+'/'+str(num_epochs)+'], Loss : '+ str(loss)
    print(lin)
    line.append(lin)



if (verbose):
    outF = open("IterativeReport.txt", "w")
    for l in line:
        # write line to output file
        outF.write(str(l))
        outF.write("\n")
    outF.close()


x = []
y = []
for i in range(len(L)):
    losss = L[i][0].detach().numpy()
    ep = L[i][1]
    x.append(ep)
    y.append(losss)

plt.scatter(x, y)
plt.savefig(resultDir + "/loss.pdf")


net.eval()
with torch.no_grad():
    x = 1
    for data in random.sample(list(test_loader), n):
        imgs, _ = data
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])
        plt.imshow(np.squeeze(outimg))
        plt.savefig(resultDir + "/" + str(x)+".pdf")
        x = x + 1

