"""Script for demonstrating linear regression with multiple variables.

Data obtained from: https://www.kaggle.com/mirichoi0218/insurance
"""

import csv
import numpy as np

NUM_ITERS = 100000
ALPHA = 0.0003

def main():
    X, y = load_data()
    theta = np.zeros((6, 1))
    print(f"Iter: {0} - Cost: {compute_cost(X, y, theta)}")
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        print(f"Iter: {i + 1} - Cost: {compute_cost(X, y, theta)}")
    while True:
        print()
        print("Medical Cost Predictor")
        y = predict(theta)
        print(f"Medical Cost: ${y}")

def load_data():
    """Loads the training set and returns it."""
    X = []
    y = []
    with open("insurance.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            X.append([
                1,
                int(row[0]),
                int(row[1] == "male"),
                float(row[2]),
                int(row[3]),
                int(row[4] == "yes"),
            ])
            y.append(float(row[6]))
    X = np.asarray(X).reshape((len(X), 6))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

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

def predict(theta):
    """Predicts y based on the user's inputs."""
    x = [
        1,
        int(input("Age: ")),
        int(input("Sex: ") == "male"),
        float(input("BMI: ")),
        int(input("Children: ")),
        int(input("Smoker: ") == "yes"),
    ]
    x = np.asarray(x).reshape((1, 6))
    y = np.matmul(x, theta)
    return np.squeeze(y)

if __name__ == "__main__":
    main()
