"""Script for demonstrating anomaly detection.

Data obtained from: https://www.kaggle.com/shelars1985/anomaly-detection-using-gaussian-distribution/data
"""

import csv
import numpy as np

EPSILON = 1e-30

def main():
    X, X_fraudulent = load_data()
    means = np.mean(X, axis=0)
    variances = np.var(X, axis=0)
    true_positives = 0
    false_positives = 0
    for x in X_fraudulent:
        if p(x, means, variances) < EPSILON:
            true_positives += 1
    for x in X:
        if p(x, means, variances) < EPSILON:
            false_positives += 1
    print(f"EPSILON = {EPSILON}")
    print(f"True Positives: {true_positives}/{X_fraudulent.shape[0]}")
    print(f"False Positives: {false_positives}/{X.shape[0]}")

def load_data():
    """Loads the training set and returns it."""
    X = []
    X_fraudulent = []
    with open("creditcard.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            if row[-1] == "0":
                X.append([float(val) for val in row[1:29]])
            else:
                X_fraudulent.append([float(val) for val in row[1:29]])
    X = np.asarray(X).reshape((len(X), 28))
    X_fraudulent = np.asarray(X_fraudulent).reshape((len(X_fraudulent), 28))
    return X, X_fraudulent

def p(x, means, variances):
    """Returns p(x) modeled by the means and variances of the features."""
    # Compute p(x) using the gaussian/normal distribution.
    return np.prod((1 / np.sqrt(2 * np.pi * variances)) *
                   np.exp(-np.square(x - means) / (2 * variances)))

if __name__ == "__main__":
    main()
