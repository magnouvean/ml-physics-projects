import numpy as np
from neuralnet import *

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1]).reshape(4, 1)

nn = NeuralNet(
    (2, 100, 100, 1),
    cost=sse,
    cost_der=sse_der,
)

nn.fit(X, y, epochs=1000, learning_rate=0.001)
print(nn.forward(X))
