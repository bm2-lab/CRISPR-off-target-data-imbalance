import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
class Smote:
    """
    SMOTE
    Parameters:
    -----------
    k: int
    sampling_rate: int
        attention sampling_rate < k.
    newindex: int

    """

    def __init__(self, sampling_rate=5, k=5):
        self.sampling_rate = sampling_rate
        self.k = k
        self.newindex = 0


    def synthetic_samples(self, X, i, k_neighbors, y=None):
        for j in range(self.sampling_rate):
            neighbor = np.random.choice(k_neighbors)
            diff = X[neighbor] - X[i]
            self.synthetic_X[self.newindex] = X[i] + random.random() * diff
            self.synthetic_y[self.newindex] = y[i] + random.random() * (y[neighbor] - y[i])
            self.newindex += 1

    def fit(self, X, y=None):
        if y is not None:
            negative_X = X[y == 0]
            X = X[y != 0]

        n_samples, n_features = X.shape
        self.synthetic_X = np.zeros((n_samples * self.sampling_rate, n_features))
        self.synthetic_y = np.zeros(n_samples * self.sampling_rate)

        knn = NearestNeighbors(n_neighbors=self.k).fit(X)
        for i in range(len(X)):
            print(i)
            k_neighbors = knn.kneighbors(X[i].reshape(1, -1),
                                         return_distance=False)[0]
            self.synthetic_samples(X, i, k_neighbors, y)

        if y is not None:
            return (np.concatenate((self.synthetic_X, X, negative_X), axis=0),
                    np.concatenate((self.synthetic_y, y[y != 0], y[y == 0]), axis=0))




def elevation_model():
    pass

