from .neuralnet import *


X = np.zeros((3, 3))
X[:, 0] = np.array([1, 2, 3])
X[:, 1] = np.array([1, 5, 3])
X[:, 2] = np.array([1, 5, 3])

nn = NeuralNet((3, 4, 2), sigmoid)
nn.forward(X)
