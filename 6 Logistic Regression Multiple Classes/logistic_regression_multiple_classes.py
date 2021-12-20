"""Script for demonstrating logistic regression with multiple classes.

Data obtained from: https://www.kaggle.com/scolianni/mnistasjpg
"""

import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

NUM_ITERS = 100
ALPHA = 1

def main():
    X, y = load_data()
    theta = np.zeros((784, 10))
    for num in range(10):  # Train a classifier for each digit.
        X_ = X
        y_ = (y == num).astype(int)
        theta_ = np.zeros((784, 1))
        print(f"{num}'s Classifier - Iter: {0} - "
              f"Cost: {compute_cost(X_, y_, theta_)}")
        for i in range(NUM_ITERS):
            theta_ = theta_ - ALPHA * compute_grad(X_, y_, theta_)
            print(f"{num}'s Classifier - Iter: {i + 1} - "
                  f"Cost: {compute_cost(X_, y_, theta_)}")
        theta[:, num] = theta_.reshape((784,))
    print(f"Training Accuracy: {compute_accuracy(X, y, theta)}")
    predict(theta)

def load_data():
    """Loads the training set and returns it."""
    X = []
    y = []
    for num in range(10):
        for file in os.listdir(f"archive/trainingSet/trainingSet/{num}/"):
            if file.endswith(".jpg"):
                image = imread(f"archive/trainingSet/trainingSet/{num}/{file}")
                image = image / 255
                image = image.reshape((1, 784))  # Flatten the image.
                X.append(image)
                y.append(num)
    X = np.asarray(X).reshape((len(X), 784))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    m = X.shape[0]
    predictions = sigmoid(np.matmul(X, theta))
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
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape((m, 1))
    acc = np.sum(predictions == y) / m
    return acc

def sigmoid(z):
    """Returns the sigmoid of a number or matrix."""
    return 1 / (1 + np.exp(-z))

def predict(theta):
    """Predicts digits in the test set using the parameters."""
    for file in os.listdir("archive/testSet/testSet/"):
        if file.endswith(".jpg"):
            image = imread(f"archive/testSet/testSet/{file}")
            plt.imshow(image)
            image = image / 255
            image = image.reshape((1, 784))
            prediction = sigmoid(np.matmul(image, theta))
            prediction = np.argmax(prediction, axis=1)
            prediction = np.squeeze(prediction)
            plt.title(f"Prediction: {prediction}")
            plt.pause(1)

if __name__ == "__main__":
    main()
