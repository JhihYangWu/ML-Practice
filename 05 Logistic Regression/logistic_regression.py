"""Script for demonstrating logistic regression.

Data obtained from: https://www.kaggle.com/kandij/diabetes-dataset
"""

import csv
import numpy as np

NUM_ITERS = 1000000
ALPHA = 0.0001

def main():
    X, y = load_data()
    theta = np.zeros((9, 1))
    print(f"Iter: {0} - Cost: {compute_cost(X, y, theta)} - "
          f"Acc: {compute_accuracy(X, y, theta)}")
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        print(f"Iter: {i + 1} - Cost: {compute_cost(X, y, theta)} - "
              f"Acc: {compute_accuracy(X, y, theta)}")
    while True:
        print()
        print("Diabetes Predictor")
        y = predict(theta)
        print(f"Diabetes Probability: {round(y * 100, 2)}%")

def load_data():
    """Loads the training set and returns it."""
    X = []
    y = []
    with open("diabetes2.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            X.append([
                1,
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                int(row[7]),
            ])
            y.append(int(row[8]))
    X = np.asarray(X).reshape((len(X), 9))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    m = X.shape[0]
    predictions = sigmoid(np.matmul(X, theta))
    # Compute the cost using logistic regression's cost function.
    J = -(1 / m) * (np.matmul(y.T, np.log(predictions)) +
                    np.matmul(1 - y.T, np.log(1 - predictions)))
    return np.squeeze(J)

def compute_grad(X, y, theta):
    """Returns the partial derivatives of the cost function."""
    m = X.shape[0]
    predictions = sigmoid(np.matmul(X, theta))
    differences = predictions - y
    grad = (1 / m) * np.matmul(X.T, differences)
    return grad

def compute_accuracy(X, y, theta):
    """Returns the accuracy of the hypothesis function."""
    m = X.shape[0]
    predictions = sigmoid(np.matmul(X, theta))
    differences = predictions - y
    acc = np.sum(np.abs(differences) < 0.5) / m
    return acc

def sigmoid(z):
    """Returns the sigmoid of a number or matrix."""
    return 1 / (1 + np.exp(-z))

def predict(theta):
    """Predicts y based on the user's inputs."""
    x = [
        1,
        int(input("Pregnancies: ")),
        int(input("Glucose: ")),
        int(input("Blood Pressure: ")),
        int(input("Skin Thickness: ")),
        int(input("Insulin: ")),
        float(input("BMI: ")),
        float(input("Diabetes Pedigree Function: ")),
        int(input("Age: ")),
    ]
    x = np.asarray(x).reshape((1, 9))
    y = sigmoid(np.matmul(x, theta))
    return np.squeeze(y)

if __name__ == "__main__":
    main()
