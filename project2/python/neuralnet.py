import typing
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    sigmoid_val = sigmoid(x)
    return sigmoid_val * (1 - sigmoid_val)


class NeuralNet:
    def __init__(
        self,
        layer_sizes: typing.Iterable,
        cost: typing.Callable,
        cost_der: typing.Callable,
        random_seed: int = 1234,
        activation_functions: typing.Iterable
        | typing.Callable[[float], float] = sigmoid,
        activation_functions_der: typing.Iterable
        | typing.Callable[[float], float] = sigmoid_der,
        output_function: typing.Callable[[float], float] = lambda x: x,
        output_function_der: typing.Callable[[float], float] = lambda _: 1,
    ):
        # Hidden layers is the sizes minus input and output layers
        self._n_hidden_layers = len(layer_sizes) - 2

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
            layer_weights = np.random.randn(layer_size, prev_layer_size) * 0.1
            layer_biases = np.zeros(layer_size).reshape((layer_size, 1)) + 0.01
            self._weights.append(layer_weights)
            self._biases.append(layer_biases)

    def _forward_one(self, X: np.array, i: int):
        """
        Forward from layer i-1 to i
        """
        return self._activation_functions[i - 1](
            self._weights[i - 1] @ X
            + np.repeat(self._biases[i - 1], X.shape[0], axis=1)
        )

    def forward(self, X: np.array):
        current_layer = self._forward_one(X, 1)
        for i in range(1, self._n_hidden_layers):
            current_layer = self._forward_one(current_layer, i + 1)
        output_layer = self._weights[-1] @ current_layer + self._biases[-1]
        return self._output_function(output_layer).T

    def fit(self, X: np.array, y: np.array, epochs: int = 100):
        for _ in range(epochs):
            current_z = self.forward(X)
            delta_L = self._output_function_der(current_z) * self._cost_function_der()
