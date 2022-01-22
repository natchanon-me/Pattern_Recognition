# JUST Plot function
import numpy as np
import matplotlib.pyplot as plt

####################################################### K-Mean Utils ###########################################################
def plot(data, set_cluster, K_cord, colors):
    plt.figure(figsize=(7,7))
    labels = [i+1 for i in range(len(colors))]
    ax = plt.subplot(1, 1, 1)
    data_x = data[:, 0]
    data_y = data[:, 1]
    K = len(K_cord)    
    
    for i in range(len(data)):
        ax.scatter(data_x[i], data_y[i], marker="o", color=colors[set_cluster[i]], s=100, edgecolors='black')
    
    for k in range(K):
        kx = K_cord[k][0]
        ky = K_cord[k][1]
        ax.scatter(kx, ky, marker="*", color=colors[k], s=400, edgecolors='black', label = labels[k])
    ax.legend()
    plt.show()

def euclid_dist(X, point_k):
    dist = np.sqrt(np.sum(np.square(X-point_k), axis=1))
    return dist


####################################################### Logistic Utils ###########################################################
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def accuracy(pred, y):
    return np.squeeze(np.squeeze((sum(y == pred)/len(y))*100))

def add_bias(X):
    Bias = np.ones((len(X), 1))
    res = np.concatenate((Bias, X), axis=1)
    return res

def random_init_param(X):
    #size of X is [m, n] where m=sample, n=features
    W = np.random.randn(len(X[0]), 1) # +1 for Bias term
    return W