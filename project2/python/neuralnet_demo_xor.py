import numpy as np

from neuralnet import *
from schedulers import *

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1]).reshape(4, 1)

scheduler = SchedulerConstant(1.0)
scheduler_adam = SchedulerAdam(1.0)
nn = NeuralNet(
    (2, 100, 100, 1),
    cost=sse,
    cost_der=sse_der,
    scheduler=scheduler,
    output_function=sigmoid,
    output_function_der=sigmoid_der,
)

nn.fit(X, y, epochs=10000, sgd=False)
print(nn.forward(X))

nn_adam = NeuralNet(
    (2, 100, 100, 1), cost=sse, cost_der=sse_der, scheduler=scheduler_adam
)

# Doesn't work with adam yet. Will be fixed soon.
nn_adam.fit(X, y, epochs=10000, sgd=False)
print(nn_adam.forward(X))
