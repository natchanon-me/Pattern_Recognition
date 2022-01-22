import numpy as np
import matplotlib.pyplot as plt
from utils import *

def Assign(data, K_cord, colors):
    all_k_dist = []
    for k in range(len(K_cord)):
        centroid = K_cord[k]
        distance = euclid_dist(data, centroid)
        all_k_dist.append(distance)

    set_cluster = np.argmin(all_k_dist, axis=0)
    plot(data, set_cluster, K_cord, colors)
    return set_cluster

def Update(data, K_cord, colors, set_cluster):
    # Update cluster centroid
    K = len(K_cord)
    for k in range(K):
        where = np.squeeze(np.argwhere(set_cluster==k))
        new_cord = np.mean(data[where], axis=0)
        K_cord[k] = new_cord
    plot(data, set_cluster, K_cord, colors)

# Cost function
def cost_function(data, K_cord, set_cluster):
    data_x = data[:, 0]
    data_y = data[:, 1]
    cost = 0
    for k in range(len(K_cord)):
        X = np.squeeze(data[np.argwhere(set_cluster==k)])
        dist = np.sqrt(np.sum(np.square(X-K_cord[k])))
        cost += dist
    return cost