import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

def cost_function(X, Y, H, W):
    # X : training data
    # H : prediction (before sigmoid)
    # Y : training label
    # W : trainable parameters
    m = len(Y)
    devide_zeros_threshold = 1e-5 # Solve devide by zero problem
    #L2 loss (IF "LINEAR")
    L2loss = (np.dot((Y-sigmoid(H)).T, (Y-sigmoid(H))))/2 # /2 for derivative term
    # or Logistic loss
    Logistloss = np.dot(-Y.T, np.log(sigmoid(H)+devide_zeros_threshold))-np.dot((1-Y).T, np.log((1-sigmoid(H))+devide_zeros_threshold))
    cost = (1/m)*Logistloss
    grad = (1/m)*(np.dot(X.T, (sigmoid(H)-Y)))
    return cost, grad

#round function uses threshold = 0.5
def predict(X, params):
    h = np.dot(X, params)
    return np.round(sigmoid(h))

def main(X, Y, W, LR):
    h = np.dot(X, W)
    cost, grad = cost_function(X, Y, h, W)
    #update parameters
    W = W - (LR)*grad
    return W, cost

X = "Your numpy array data"
Y = "Your numpy array label"
# Add bias for training sample
X = add_bias(X)
W = random_init_param(X)
W_lowest = np.zeros((len(X[0]), 1))

LR = 0.001
epoch = 100000
best_loss = 1e6 


for i in range(epoch+1):
    W, cost = main(X, Y, W)
    pred = predict(X, W)
    if cost < best_loss:
        W_lowest = W
    if (i%1000 == 0): # Just for logging
        print("Epoch:", i+1) 
        acc = accuracy(pred, Y)
        print("Logistic loss : ", np.squeeze(cost))
        print("Training accuracy = {:.2f}%\n".format(acc))