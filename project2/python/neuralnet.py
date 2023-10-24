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


def sse(y_pred, y):
    diff = y - y_pred
    return (diff.T @ diff)[0, 0]


def sse_der(y_pred, y):
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
        lmbda: float = 0.0,  # penalization isn't really implemented yet, so this is somewhat useless for now
    ):
        # Hidden layers is the sizes minus input and output layers
        self._n_hidden_layers = len(layer_sizes) - 2
        # Set hyperparameters
        self._lmbda = lmbda

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

    def fit(
        self,
        X: np.array,
        y: np.array,
        epochs: int = 100,
        learning_rate=0.1,
        sgd=False,
        sgd_size=10,
    ):
        """Fit the neural network using backpropagation as described in goodfellow et al 6.4 algorithm

        Args:
            X (np.array): Design matrix
            y (np.array): Response/target
            epochs (int, optional): Number of epochs to run. Defaults to 100.
            learning_rate (float, optional): The size of the learning rate parameter. Defaults to 0.1.
        """
        for _ in range(epochs):
            rand_indices = np.random.choice(X.shape[0], sgd_size, replace=False)
            X_data = X[rand_indices, :] if sgd else X
            y_data = y[rand_indices] if sgd else y
            # We start with finding a in the L layer (also known as the target)
            # by doing a forward propagation
            y_pred = self.forward(X_data)
            print(f"COST: {self._cost_function(y_pred, y_data)}")

            weight_grads = [0] * (self._n_hidden_layers + 1)
            bias_grads = [0] * (self._n_hidden_layers + 1)

            g = self._cost_function_der(y_pred, y_data)
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
                weight_grads[k - 1] = h_k_1.T @ g
                bias_grads[k - 1] = np.sum(g, axis=0)
                g = g @ W_k.T

            for k, (weight_grad, bias_grad) in enumerate(zip(weight_grads, bias_grads)):
                self._weights[k] += learning_rate * weight_grad
                self._biases[k] += learning_rate * bias_grad
