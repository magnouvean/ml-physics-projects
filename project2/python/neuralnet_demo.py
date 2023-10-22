from neuralnet import *

n = 7
X = np.zeros((7, 2))
Y = np.zeros_like(X)
x_1 = np.linspace(-1, 1, n)
x_2 = np.linspace(0, 1, n)
X[:, 0] = (x_1 - np.mean(x_1)) / np.std(x_1)
X[:, 1] = x_2 - np.mean(x_2) / np.std(x_2)


def some_func(x_1, x_2):
    return 2 * x_1 * x_2, x_1**2


y = (x_1**2 + 2 * x_2).reshape(n, 1)

nn = NeuralNet(
    (2, 10, 10, 1),
    cost=sse,
    cost_der=sse_der,
)
nn.fit(X, y, learning_rate=0.1, epochs=1000)
my_y_pred = nn.forward(X)
print(f"mse (personal): {mse(y, my_y_pred)}")

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100, 100))
model.fit(X, y)
sklearn_y_pred = model.predict(X).reshape(n, 1)
print(f"mse (sklearn): {mse(y, sklearn_y_pred)}")
