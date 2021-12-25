"""Script for demonstrating the K-Means Clustering Algorithm."""

import numpy as np
import matplotlib.pyplot as plt
import random

K = 4  # Number of clusters.
NUM_ITERS = 5
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

def main():
    X = create_data()
    M = initialize_centroids(X)
    for i in range(NUM_ITERS):
        c = assign_cluster(X, M)
        plot(X, c, M)
        M = move_centroids(M, X, c)
        plot(X, c, M)
    print(f"Cost (rerun if too high): {compute_cost(X, c, M)}")
    plt.show()

def create_data():
    """Creates a training set and returns it."""
    X = [
        [1.4369978915091262, 0.5459313252206397],
        [0.861793080113828, 1.9554060835324798],
        [1.0025808515708865, 0.8783289651070403],
        [0.8902387407096419, 1.4631443634978936],
        [1.0346150128042528, 1.9339876188701008],
        [1.3869103596276016, 1.669044403169197],
        [0.7546696297241212, 0.11823610115736694],
        [1.195398322792624, 1.7885275544370849],
        [0.7644722719356507, 1.7410437218694441],
        [1.404034691347272, 0.1331522977612365],
        [1.4238079031785944, 3.179161361841972],
        [0.633101614645742, 3.0876046005741196],
        [1.9512373034247181, 3.300036444343764],
        [1.862565709727386, 2.5444169660863243],
        [0.645597008939107, 2.5670578129936072],
        [0.411848884989938, 3.149154802511826],
        [1.7920677617913336, 2.696648071089473],
        [1.7382711715708765, 2.9513677238124476],
        [3.623515055339289, 1.8949532173002193],
        [4.300002086333069, 1.5809176856966434],
        [4.237454763839434, 1.2625753902447496],
        [4.049955712139477, 1.8182829126399358],
        [4.791528484283601, 1.6968164794434129],
        [4.363669328579114, 1.7485233994591816],
        [3.510758195007222, 1.3734588535839847],
        [3.2228545485499964, 1.423086509852012],
        [4.990950842836408, 2.12198166177197],
        [4.726481164584278, 1.0299518070204574],
        [3.665133389246348, 2.639988170376059],
        [3.0081677274004006, 1.5836249815821626],
    ]
    X = np.asarray(X).reshape((len(X), 2))
    return X

def initialize_centroids(X):
    """Returns a matrix with K randomly initialized cluster centroids."""
    m, n = X.shape
    M = np.zeros((n, K))
    sample = random.sample(range(m), K)
    for i in range(K):
        # Initialize the centroids to randomly selected training examples.
        M[:, i] = X[sample[i]]
    return M

def assign_cluster(X, M):
    """Assigns each training example to the centroid it is closest to."""
    m, n = X.shape
    c = np.zeros((m, 1))
    for i in range(m):
        xi = X[i].reshape((n, 1))
        c[i] = np.argmin(np.sum(np.square(xi - M), axis=0))
    return c

def move_centroids(M, X, c):
    """Moves the centroids to the middle of the examples assigned to it."""
    for i in range(K):
        temp = (c == i).astype(int)
        M[:, i] = (1 / np.sum(temp)) * np.matmul(temp.T, X)
    return M

def plot(X, c, M):
    """Plots the training set and the cluster centroids."""
    m = X.shape[0]
    plt.clf()
    # Plot the training set.
    for i in range(m):
        plt.scatter(X[i][0], X[i][1], marker="o", c=COLORS[int(c[i])])
    # Plot the cluster centroids.
    for i in range(K):
        plt.scatter(M[0][i], M[1][i], marker="x", c=COLORS[i], s=100)
    plt.pause(0.5)

def compute_cost(X, c, M):
    """Returns the cost based on the clusters it created."""
    m, n = X.shape
    J = 0
    for i in range(m):
        difference = X[i] - M[:, int(c[i])]
        difference = difference.reshape((n, 1))
        J += np.matmul(difference.T, difference)
    J *= 1 / m
    return np.squeeze(J)

if __name__ == "__main__":
    main()
