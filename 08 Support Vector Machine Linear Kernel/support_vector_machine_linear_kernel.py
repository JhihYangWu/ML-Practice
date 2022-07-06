"""Script for demonstrating support vector machine with a linear kernel."""

import numpy as np
import matplotlib.pyplot as plt

NUM_ITERS = 600
ALPHA = 0.0003
C = 10  # Regularization parameter.
EPSILON = 1e-4

def main():
    X, y = create_data()
    theta = np.zeros((3, 1))
    plot(X, y, theta, 0)
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        plot(X, y, theta, i + 1)
    plt.show()

def create_data():
    """Creates a training set and returns it."""
    X = [
        [1, 0.03363841641793974, 1.5309938142513888],
        [1, 0.6352462668454155, 0.8611894194305876],
        [1, 1.5833609229028835, 0.696571792644405],
        [1, 0.2275643691739857, 1.062276775800881],
        [1, 1.5129239963550993, 0.7213727731089379],
        [1, 0.39191791752409677, 1.4443274835385764],
        [1, 0.07183595741205795, 0.7550219064477008],
        [1, 0.4058303242747474, 0.7550091917519719],
        [1, 1.1162522638575245, 2.2749465243726403],
        [1, 2.1864279523766363, 1.9377090329935227],
        [1, 2.4775493050957205, 1.1529869936626722],
        [1, 1.4363247065906892, 2.8020469469260414],
        [1, 2.4279509108782857, 1.541722581625922],
        [1, 2.889837408734383, 2.6509325851955765],
        [1, 1.6676972861801, 1.4634767175556471],
        [1, 2.339860280289113, 2.941955350546089],
        [1, 1.5421631455089195, 1.6311661408764375],
        [1, 2.522405530223356, 2.972381860797738],
        [1, 1.218415004411893, 2.2682284954836023],
        [1, 2.874403688113408, 1.4317238281047244],
    ]
    y = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X = np.asarray(X).reshape((len(X), 3))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    predictions = np.matmul(X, theta)
    temp = theta[1:]
    J = C * (np.matmul(y.T, cost1(predictions)) + np.matmul(1 - y.T,
        cost0(predictions))) + 0.5 * np.matmul(temp.T, temp)
    return np.squeeze(J)

def cost1(z):
    """Returns the cost of z when y = 1."""
    return np.maximum(0, -(z - 1))

def cost0(z):
    """Returns the cost of z when y = 0."""
    return np.maximum(0, z + 1)

def compute_grad(X, y, theta):
    """Returns the partial derivatives of the cost function."""
    grad = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        # Approximate the gradient numerically.
        temp1 = np.copy(theta)
        temp2 = np.copy(theta)
        temp1[i] += EPSILON
        temp2[i] -= EPSILON
        loss1 = compute_cost(X, y, temp1)
        loss2 = compute_cost(X, y, temp2)
        grad[i] = (loss1 - loss2) / (2 * EPSILON)
    return grad

def plot(X, y, theta, iter):
    """Plots the training set and the line."""
    m = X.shape[0]
    plt.clf()
    plt.title(f"Iter: {iter} - Cost: {compute_cost(X, y, theta)}")
    # Plot the X's and O's.
    for i in range(m):
        plt.scatter(X[i][1], X[i][2], marker="x" if y[i] else "o",
                    c="r" if y[i] else "b")
    # Plot the decision boundary.
    plt.plot([0, 3],
             [-theta[0] / theta[2], (-theta[0] - theta[1] * 3) / theta[2]])
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.pause(0.01)

if __name__ == "__main__":
    main()
