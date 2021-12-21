"""Script for demonstrating regularization.

Data obtained from: https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 10000  # Regularization parameter.

def main():
    X, y = load_data()
    # Calculate theta using the regularized normal equation.
    L = np.identity(10)
    L[0][0] = 0
    theta = np.matmul(np.linalg.inv(np.matmul(X.T, X) + LAMBDA * L),
                      np.matmul(X.T, y))
    plot(X, y, theta)

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

def predict(num_years, theta):
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
    y = np.matmul(x, theta)
    return np.squeeze(y)

def plot(X, y, theta):
    """Plots the training set and the line."""
    plt.scatter(X[:, 1], y)
    x_values = [i / 10 for i in range(115)]
    y_values = [predict(x, theta) for x in x_values]
    plt.plot(x_values, y_values)
    plt.xlim(0, 11.5)
    plt.ylim(0, 150000)
    plt.title(f"LAMBDA = {LAMBDA}")
    plt.show()

if __name__ == "__main__":
    main()
