import copy
import typing

import numpy as np

from schedulers import Scheduler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    sigmoid_val = sigmoid(x)
    return sigmoid_val * (1 - sigmoid_val)


def relu(x):
    return (x > 0) * x


def relu_der(x):
    return (x > 0) * 1


def sse(y_pred, y):
    diff = y - y_pred
    return (diff.T @ diff)[0, 0]


def sse_der(y_pred, y):
    return 2 * (y_pred - y)


def mse(y_pred, y):
    return sse(y, y_pred) / len(y)


def l2regularizer(grad: np.ndarray) -> np.ndarray:
    return grad


class NeuralNet:
    def __init__(
        self,
        layer_sizes: typing.Iterable,
        cost: typing.Callable,
        cost_der: typing.Callable,
        scheduler: Scheduler,
        random_seed: int = 1234,
        activation_functions: typing.Iterable
        | typing.Callable[[float], float] = sigmoid,
        activation_functions_der: typing.Iterable
        | typing.Callable[[float], float] = sigmoid_der,
        output_function: typing.Callable[[float], float] = lambda x: x,
        output_function_der: typing.Callable[[float], float] = lambda _: 1,
        lmbda: float = 0.0,  # penalization isn't really implemented yet, so this is somewhat useless for now
        regularizer: typing.Callable[[np.ndarray], np.ndarray] = l2regularizer,
    ):
        # Hidden layers is the sizes minus input and output layers
        self._n_hidden_layers = len(layer_sizes) - 2
        # Penalization variables
        self._lmbda = lmbda
        self._regularizer = regularizer

        self._scheduler_weights = [
            copy.deepcopy(scheduler) for _ in range(self._n_hidden_layers + 1)
        ]
        self._scheduler_biases = [
            copy.deepcopy(scheduler) for _ in range(self._n_hidden_layers + 1)
        ]

        # Error handling
        if self._n_hidden_layers == 0:
            raise ValueError("Must have at least one hidden layer")
        elif not (
            callable(activation_functions) and callable(activation_functions_der)
        ) and len(activation_functions) != len(activation_functions_der):
            raise ValueError(
                "Activation function list and its derivatives does not match dimensions"
            )

        self._cost_function = cost
        self._cost_function_der = cost_der

        # Set activation and output functions with derivatives
        if callable(activation_functions):
            self._activation_functions = [activation_functions] * self._n_hidden_layers
        else:
            self._activation_functions = activation_functions
        if callable(activation_functions_der):
            self._activation_functions_der = [
                activation_functions_der
            ] * self._n_hidden_layers
        else:
            self._activation_functions_der = activation_functions_der
        self._output_function = output_function
        self._output_function_der = output_function_der

        # a-s are the values of the layers, while z-s are the values of the layers before activation
        self._h = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]
        self._a = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]

        # Initialize weights/biases
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self._weights = []
        self._biases = []
        for i, layer_size in enumerate(layer_sizes[1:]):
            # Since we start enumerating through the list from the first
            # element, i will always be one less than the index of the
            # corresponding layer_size, so i is the index of the previous layer.
            prev_layer_size = layer_sizes[i]
            layer_weights = np.random.randn(prev_layer_size, layer_size) * 0.00001
            layer_biases = np.zeros(layer_size)
            self._weights.append(layer_weights)
            self._biases.append(layer_biases)

    def _forward_one(self, X: np.array, i: int):
        """
        Forward to hidden layer `i` (or the output layer if `i` = `number of
        hidden layer plus 1`)
        """
        activation_func = (
            self._activation_functions[i - 1]
            if i <= self._n_hidden_layers
            else self._output_function
        )
        z = X @ self._weights[i - 1]
        for j in range(z.shape[0]):
            z[j, :] += self._biases[i - 1]
        a = activation_func(z)

        return z, a

    def forward(self, X: np.array):
        self._h_0 = X
        h_prev = self._h_0
        for i in range(self._n_hidden_layers + 1):
            self._a[i], self._h[i] = self._forward_one(h_prev, i + 1)
            h_prev = self._h[i]

        return self._h[-1]

    def _print_performance(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.forward(X)
        print(f"COST: {self._cost_function(y_pred, y)}")

    def _backward_once(self, X: np.ndarray, y: np.ndarray):
        """Performs one backward propagation on some data.

        Args:
            X (np.ndarray): The design matrix of the data to use.
            y (np.ndarray): The response of the data to use.
        """
        # We start with finding a in the L layer (also known as the target)
        # by doing a forward propagation
        y_pred = self.forward(X)

        weight_grads = [0] * (self._n_hidden_layers + 1)
        bias_grads = [0] * (self._n_hidden_layers + 1)

        g = self._cost_function_der(y_pred, y)
        for k in range(self._n_hidden_layers + 1, 0, -1):
            h_k_1 = self._h[k - 2] if k > 1 else self._h_0
            a_k = self._a[k - 1]
            W_k = self._weights[k - 1]
            f_der = (
                self._output_function_der
                if k == self._n_hidden_layers + 1
                else self._activation_functions_der[k - 1]
            )
            g = g * f_der(a_k)
            weight_grads[k - 1] = h_k_1.T @ g + self._lmbda * self._regularizer(W_k)
            bias_grads[k - 1] = np.sum(g, axis=0)
            g = g @ W_k.T

        for k, (weight_grad, bias_grad) in enumerate(zip(weight_grads, bias_grads)):
            weight_update = self._scheduler_weights[k].update(weight_grad)
            bias_update = self._scheduler_biases[k].update(bias_grad)
            self._weights[k] += weight_update
            self._biases[k] += bias_update

    def fit(
        self,
        X: np.array,
        y: np.array,
        epochs: int = 200,
        sgd=False,
        sgd_size=10,
        print_every: int = 0,
    ):
        """Train the neural network to some data

        Args:
            X (np.array): The design matrix.
            y (np.array): The response. Must be of compatible dimensions as X.
            epochs (int, optional): The max amount of epochs to allow. Defaults to 100.
            sgd (bool, optional): Wether to use stochastic gradient descent or not. Defaults to False.
            sgd_size (int, optional): The size of each mini-batch when using stochastic gradient descent. Defaults to 10.
            print_every (int, optional): Print the cost every `print_every` epoch. If 0 do not print at all. Defaults to 0.
        """
        # Reset all the schedulers
        for k in range(self._n_hidden_layers + 1):
            self._scheduler_weights[k].reset()
            self._scheduler_biases[k].reset()

        # Fail if the sgd size is too big for the data
        if X.shape[0] < sgd_size:
            raise ValueError(
                f"The size of a minibatch in stochastic gradient descent cannot be bigger than the amount of datapoints (datapoints: {X.shape[0]}, sgd_size: {sgd_size})"
            )

        for i in range(epochs):
            n_iter_per_epoch = 1 if not sgd else int(X.shape[0] / sgd_size)
            for _ in range(n_iter_per_epoch):
                if sgd:
                    rand_indices = np.random.choice(X.shape[0], sgd_size, replace=False)
                X_data = X[rand_indices, :] if sgd else X
                y_data = y[rand_indices] if sgd else y
                self._backward_once(X_data, y_data)

            if print_every > 0 and i % print_every == 0:
                self._print_performance(X, y)
