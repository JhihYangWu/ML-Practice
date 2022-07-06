"""Script for demonstrating support vector machine with a gaussian kernel."""

import numpy as np
import matplotlib.pyplot as plt

NUM_ITERS = 400
ALPHA = 0.0001
C = 100
EPSILON = 1e-4
SIGMA_SQUARED = 0.1  # Gaussian kernel's parameter.

def main():
    X, y = create_data()
    m = X.shape[0]
    theta = np.zeros((m + 1, 1))
    print(f"Iter: {0} - Cost: {compute_cost(X, y, theta)}")
    for i in range(NUM_ITERS):
        theta = theta - ALPHA * compute_grad(X, y, theta)
        print(f"Iter: {i + 1} - Cost: {compute_cost(X, y, theta)}")
    plot(X, y, theta)

def create_data():
    """Creates a training set and returns it."""
    X = [
        [1.3982085446208352, 2.895791883385896],
        [2.7128990837805658, 1.709305044103482],
        [1.7932887771460444, 2.7120055404354813],
        [1.7587518292943858, 2.20167813681815],
        [1.3433776605023853, 2.5540043987357417],
        [2.35960613911514, 1.9338920421648456],
        [2.285291164627441, 1.6844906931397212],
        [1.0234770505075863, 2.9163097661859316],
        [1.4759219170542943, 2.830120614756429],
        [2.4793471751766827, 2.6869929807261146],
        [1.9925941190570582, 1.1966777713473407],
        [2.2021375650179973, 2.732208086452098],
        [2.3018243284580793, 1.9416690308970792],
        [1.9875755619751019, 2.511979886884702],
        [1.0601940366143288, 1.873353921471216],
        [2.870418206151869, 2.3274096427622606],
        [2.051868235525511, 0.29360130905568216],
        [3.701638473815533, 3.947328231153682],
        [0.3165857087841992, 0.877055072536169],
        [1.345954677199066, 3.176255957777928],
        [0.7652215691641349, 0.37160122694098163],
        [1.4809683789147043, 0.24599676179195606],
        [3.4425653179476616, 1.6572722050435713],
        [1.0388743798060562, 3.600853868450417],
        [2.636613997434798, 3.8063739181906664],
        [3.979974342000413, 3.671820447903287],
        [3.9891545474750747, 3.5996948587996194],
        [1.4075532533662258, 3.651157759656933],
        [0.9011797781623145, 0.194523499659363],
        [0.9242683047018172, 1.354078326909859],
        [0.38980362897564236, 0.6284042632945832],
        [3.593855016570258, 3.772469654637333],
        [2.0421469946021396, 3.2741446561122567],
        [1.0298622552109191, 0.6661092285350305],
        [2.5817985463692827, 0.3200655316839782],
        [3.171242804490441, 1.3776269433218027],
        [2.9372455156425374, 0.7851858618898158],
        [3.9092319083286684, 1.8082266631041564],
        [2.468860936044197, 0.7847805312312692],
        [0.07580062493784201, 0.9080232179294887],
        [3.523154422179173, 0.4748501636596907],
        [0.8533237876047526, 2.2723151890728683],
        [3.7401812592514387, 1.956210758807968],
        [3.346525993512851, 3.54811871515001],
        [1.9868411128399863, 0.8464191829875012],
        [2.549533046134562, 3.8328254502516046],
        [3.714468367113781, 3.124476973762224],
        [1.2356688778414129, 3.760319401931582],
        [3.739814954164196, 1.6458810240911173],
        [0.28346100599914914, 3.1288236745168083],
        [3.772266047002858, 2.2265179756841538],
        [3.452494789445582, 1.7007430891788156],
        [2.790532869828955, 2.9356077832045653],
        [1.155890405842431, 1.1110343857509415],
        [3.826239429804105, 0.5181327997399809],
    ]
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0]
    X = np.asarray(X).reshape((len(X), 2))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y

def compute_cost(X, y, theta):
    """Returns the cost for being parameterized by theta."""
    m = X.shape[0]
    # Convert X into featurized F.
    F = np.zeros((m, m + 1))
    for i in range(F.shape[0]):
        F[i] = get_features(X[i].T, X).reshape((m + 1,))
    predictions = np.matmul(F, theta)
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

def get_features(x, X):
    """Transforms an x vector into an m + 1 dimensional feature vector."""
    m = X.shape[0]
    x = x.reshape((x.shape[0], 1))
    # Compute f using gaussian kernel.
    f = np.exp(-np.sum(np.square(x - X.T), axis=0) / (2 * SIGMA_SQUARED))
    # Append a 1 on the left of f because f0 = 1.
    f = np.append(1, f)
    f = f.reshape((m + 1, 1))
    return f

def compute_grad(X, y, theta):
    """Returns the partial derivatives of the cost function."""
    grad = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        temp1 = np.copy(theta)
        temp2 = np.copy(theta)
        temp1[i] += EPSILON
        temp2[i] -= EPSILON
        loss1 = compute_cost(X, y, temp1)
        loss2 = compute_cost(X, y, temp2)
        grad[i] = (loss1 - loss2) / (2 * EPSILON)
    return grad

def plot(X, y, theta):
    """Plots the training set and the non-linear decision boundary."""
    m = X.shape[0]
    # Plot the non-linear decision boundary using colored dots.
    for i in range(41):
        for j in range(41):
            x = [i / 10, j / 10]
            x = np.asarray(x).reshape((2, 1))
            f = get_features(x, X)
            prediction = int(np.matmul(f.T, theta) >= 0)
            plt.scatter(x[0], x[1], c="r" if prediction else "b", s=1)
    # Plot the training set using X's and O's.
    for i in range(m):
        plt.scatter(X[i][0], X[i][1], marker="x" if y[i] else "o",
                    c="r" if y[i] else "b")
    plt.show()

if __name__ == "__main__":
    main()
