import numpy as np
import torch


df = np.genfromtxt('./HW2/even_mnist.csv',delimiter =' ')

class Data():
    def __init__(self):

        x_train = []
        y_train = []

        x_test = []
        y_test = []

        for i in range(29492):
            if i <= 29192:
                x_train_i = []
                y_train.append(df[i][196])
                for j in range(196):
                    x_train_i.append(df[i][j])

                x_train.append(x_train_i)

            else:
                x_test_i = []
                y_test.append(df[i][196])
                for j in range(196):
                    x_test_i.append(df[i][j])

                x_test.append(x_test_i)

        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        x_test = torch.tensor(x_test)
        y_test = torch.tensor(y_test)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

