import typing
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    sigmoid_val = sigmoid(x)
    return sigmoid_val * (1 - sigmoid_val)


def relu(x):
    return (x > 0) * x


def relu_der(x):
    return (x > 0) * 1


def sse(y, y_pred):
    diff = y - y_pred
    return (diff.T @ diff)[0, 0]


def sse_der(y, y_pred):
    return 2 * (y - y_pred)


def mse(y, y_pred):
    return sse(y, y_pred) / len(y)


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

        # a-s are the values of the layers, while z-s are the values of the layers before activation
        self._a = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]
        self._z = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]

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
            layer_biases = np.zeros(layer_size) + 0.00001
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
            z[j, :] = self._biases[i - 1]
        a = activation_func(z)

        return z, a

    def forward(self, X: np.array):
        self._a_0 = X
        self._z[0], self._a[0] = self._forward_one(X, 1)
        for i in range(1, self._n_hidden_layers + 1):
            self._z[i], self._a[i] = self._forward_one(self._a[i - 1], i + 1)

        return self._a[-1]

    def fit(self, X: np.array, y: np.array, epochs: int = 100, learning_rate=0.1):
        for _ in range(epochs):
            # We start with finding a in the L layer (also known as the target)
            # by doing a forward propagation
            self.forward(X)
            print(f"COST: {self._cost_function(y, self._a[-1])}")

            weight_updates = [0] * (self._n_hidden_layers + 1)
            bias_updates = [0] * (self._n_hidden_layers + 1)

            delta_l = self._output_function_der(self._z[-1]) * self._cost_function_der(
                y, self._a[-1]
            )
            w_grad_l = self._a[-2].T @ delta_l
            b_grad_l = np.sum(delta_l, axis=0)

            weight_updates[-1] = -learning_rate * w_grad_l
            bias_updates[-1] = -learning_rate * b_grad_l

            for l in range(self._n_hidden_layers - 1, -1, -1):
                a_prev = self._a[l - 1] if l > 0 else self._a_0
                delta_l = (
                    delta_l
                    @ self._weights[l + 1].T
                    * self._activation_functions_der[l](self._z[l])
                )
                w_grad_l = a_prev.T @ delta_l
                b_grad_l = np.sum(delta_l, axis=0)
                weight_updates[l] = -learning_rate * w_grad_l
                bias_updates[l] = -learning_rate * b_grad_l

            for l, (weight_update, bias_update) in enumerate(
                zip(weight_updates, bias_updates)
            ):
                self._weights[l] -= weight_update
                self._biases[l] -= bias_update
