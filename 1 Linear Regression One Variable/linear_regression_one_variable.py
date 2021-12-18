"""Script for demonstrating linear regression with one variable.

Data obtained from: https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

NUM_ITERS = 20
ALPHA = 0.01  # Learning rate.

def main():
    X, y = load_data()
    theta = np.zeros((2, 1))  # Parameters.
    plot(X, y, theta, 0)
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        plot(X, y, theta, i + 1)
    plt.show()

def load_data():
    """Loads the training set and returns it."""
    X = []
    y = []
    with open("Salary_Data.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row.
        for row in csv_reader:
            X.append([1, float(row[0])])
            y.append(float(row[1]))
    X = np.asarray(X).reshape((len(X), 2))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    m = X.shape[0]  # Number of training examples.
    differences = np.matmul(X, theta) - y
    J = (1 / (2 * m)) * np.matmul(differences.T, differences)
    return np.squeeze(J)

def compute_grad(X, y, theta):
    """Returns the partial derivatives of the cost function."""
    m = X.shape[0]
    differences = np.matmul(X, theta) - y
    grad = (1 / m) * np.matmul(X.T, differences)
    return grad

def predict(x, theta):
    """Predicts y given x and the parameters."""
    y = theta[0][0] + theta[1][0] * x
    return y

def plot(X, y, theta, iter):
    """Plots the training set and the line."""
    plt.clf()
    plt.title(f"Iter: {iter} - Cost: {compute_cost(X, y, theta)}")
    plt.scatter(X[:, 1], y)
    plt.plot([0, 11.5], [predict(0, theta), predict(11.5, theta)])
    plt.pause(0.5)

if __name__ == "__main__":
    main()
