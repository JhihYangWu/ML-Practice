"""Script for demonstrating Principal Component Analysis (PCA).

Data obtained from: https://www.kaggle.com/dheeraj07/principal-component-analysis
"""

import csv
import numpy as np

k = 2  # Number of Principal Components.

def main():
    X = load_data()
    m = X.shape[0]
    Sigma = (1 / m) * np.matmul(X.T, X)  # Covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    # Find k cols with the largest eigenvalues.
    k_indices = np.argsort(eigenvalues)[::-1][:k]
    U_reduce = eigenvectors[:, k_indices]
    print("X = ")
    np.set_printoptions(3, suppress=True)
    print(X)
    print("Z = ")
    Z = np.matmul(X, U_reduce)
    print(Z)
    print(f"k = {k}")
    variance_retained = 1 - (compute_projection_error(X, Z, U_reduce) /
                             compute_total_variation(X))
    print(f"Variance Retained: {round(variance_retained * 100, 2)}%")

def load_data():
    """Loads the training set and returns it."""
    X = []
    with open("Longley (1).csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            X.append([float(val) for val in row])
    X = np.asarray(X).reshape((len(X), 6))
    return X - np.mean(X, axis=0)  # Return mean normalized version of X.

def compute_projection_error(X, Z, U_reduce):
    """Returns the average squared projection error."""
    m = X.shape[0]
    X_approx = np.matmul(Z, U_reduce.T)  # Reconstruct X using Z and U_reduce.
    projection_error = (1 / m) * np.sum(np.square(X - X_approx))
    return projection_error

def compute_total_variation(X):
    """Returns the total variation in the dataset."""
    m = X.shape[0]
    total_variation = (1 / m) * np.sum(np.square(X))
    return total_variation

if __name__ == "__main__":
    main()
