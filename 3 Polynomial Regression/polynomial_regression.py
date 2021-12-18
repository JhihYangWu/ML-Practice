"""Script for demonstrating polynomial regression.

Data obtained from: https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

NUM_ITERS = 250
ALPHA = 0.01

def main():
    X, y = load_data()
    X, stds = scale_features(X)
    theta = np.zeros((10, 1))
    plot(X, y, theta, 0, stds)
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        plot(X, y, theta, i + 1, stds)
    plt.show()

def load_data():
    """Loads the training set and returns it."""
    X = []
    y = []
    with open("Salary_Data.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            X.append([
                1,
                float(row[0]),
                float(row[0]) ** 2,
                float(row[0]) ** 3,
                float(row[0]) ** 4,
                float(row[0]) ** 5,
                float(row[0]) ** 6,
                float(row[0]) ** 7,
                float(row[0]) ** 8,
                float(row[0]) ** 9,
            ])
            y.append(float(row[1]))
    X = np.asarray(X).reshape((len(X), 10))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def scale_features(X):
    """Scales the columns of X to have 1 std and returns it."""
    stds = np.std(X, axis=0)
    stds[0] = 1  # Make sure feature scaling isn't applied to x0.
    scaled_X = X / stds
    return scaled_X, stds

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    m = X.shape[0]
    differences = np.matmul(X, theta) - y
    J = (1 / (2 * m)) * np.matmul(differences.T, differences)
    return np.squeeze(J)

def compute_grad(X, y, theta):
    """Returns the partial derivatives of the cost function."""
    m = X.shape[0]
    differences = np.matmul(X, theta) - y
    grad = (1 / m) * np.matmul(X.T, differences)
    return grad

def predict(num_years, theta, stds):
    """Predicts salary given num_years of experience using the parameters."""
    x = [
        1,
        num_years,
        num_years ** 2,
        num_years ** 3,
        num_years ** 4,
        num_years ** 5,
        num_years ** 6,
        num_years ** 7,
        num_years ** 8,
        num_years ** 9,
    ]
    x = np.asarray(x).reshape((1, 10))
    scaled_x = x / stds
    y = np.matmul(scaled_x, theta)
    return np.squeeze(y)

def plot(X, y, theta, iter, stds):
    """Plots the training set and the line."""
    plt.clf()
    plt.title(f"Iter: {iter} - Cost: {compute_cost(X, y, theta)}")
    plt.scatter(X[:, 1] * stds[1], y)
    x_values = [i / 10 for i in range(115)]
    y_values = [predict(x, theta, stds) for x in x_values]
    plt.plot(x_values, y_values)  # Plot the non-linear line.
    plt.xlim(0, 11.5)
    plt.ylim(0, 150000)
    plt.pause(0.1)

if __name__ == "__main__":
    main()
