import numpy as np

from neuralnet import *
from schedulers import *

np.random.seed(314159)


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


n = 700
X = np.zeros((n, 2))
Y = np.zeros_like(X)
x_1 = np.random.uniform(0, 1, n)
x_2 = np.random.uniform(0, 1, n)
X[:, 0] = (x_1 - np.mean(x_1)) / np.std(x_1)
X[:, 1] = x_2 - np.mean(x_2) / np.std(x_2)
y = franke_function(x_1, x_2).reshape(n, 1)


# Define the schedulers we need:
scheduler_constant = SchedulerConstant(learning_rate=0.0005)
scheduler_rmsprop = SchedulerRMSProp(learning_rate=0.005)
scheduler_adagrad = SchedulerAdagrad(learning_rate=0.05)
scheduler_adam = SchedulerAdam(learning_rate=0.05)

n_epochs = 20

for method_name, scheduler in [
    ("Constant", scheduler_constant),
    ("RMSProp", scheduler_rmsprop),
    ("Adagrad", scheduler_adagrad),
    ("Adam", scheduler_adam),
]:
    nn = NeuralNet(
        (2, 10, 1),
        scheduler=scheduler,
        cost=sse,
        cost_der=sse_der,
    )
    nn.fit(X, y, epochs=n_epochs, print_every=20, sgd=True, sgd_size=32)
    my_y_pred = nn.forward(X)
    print(f"mse ({method_name}): {mse(y, my_y_pred)}")

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=0)
model.fit(X, y)
sklearn_y_pred = model.predict(X).reshape(n, 1)
print(f"mse (sklearn): {mse(y, sklearn_y_pred)}")
