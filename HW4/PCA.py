import numpy as np
import os

class PCA():
    def __init__(self, n_component=None):
        self.n_component = n_component
        self.eig_vector = None
    
    def mean(self, X):
        self.means = np.mean(X, axis=1, keepdims=True)
        return self.means

    def get_gram_eigen(self, X_no_mean):
        cov = np.dot(X_no_mean, X_no_mean.T)
        D, Q = np.linalg.eigh(cov)
        ind = np.argsort(D)
        D = np.flip(np.array(D)[ind])
        Q = np.flip(np.array(Q)[:][ind], axis=1)
        return D, Q

    def fit(self, X):
        # X(N, D): N is number of example, D is number of features.
        X = X.T
        x_mean = self.mean(X)
        X_tilda = X - x_mean
        D, Q = self.get_gram_eigen(X_no_mean = X_tilda)
        Q = Q/np.linalg.norm(Q, axis=0)
        return Q
    
    def transform(self, X, eigenProj):
        return np.dot(X, eigenProj[:, :self.n_component])

    def reconstruct(self, X_proj, eigenProj):
        rec = np.sum(eigenProj[:,:self.n_component]*X_proj.T, axis=1, keepdims=True) + self.means
        return rec
