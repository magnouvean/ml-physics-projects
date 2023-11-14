import copy
import typing

import jax
import numpy as np

from .functions import *
from .schedulers import Scheduler


class NeuralNet:
    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        cost: typing.Callable[[np.ndarray, np.ndarray], float],
        scheduler: Scheduler,
        random_seed: int = 1234,
        cost_grad: typing.Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        activation_functions: list[typing.Callable[[np.ndarray], np.ndarray]]
        | typing.Callable[[np.ndarray], np.ndarray] = sigmoid,
        activation_functions_der: typing.Callable[[np.ndarray], np.ndarray]
        | None = None,
        output_function: typing.Callable[
            [np.ndarray], np.ndarray
        ] = identity_output_function,
        output_function_der: typing.Callable[[np.ndarray], np.ndarray] | None = None,
        lmbda: float = 0.0,
        regularizer: typing.Callable[[np.ndarray], np.ndarray] = l2regularizer,
        regularizer_der: typing.Callable[[np.ndarray], np.ndarray] = l2regularizer_der,
        biases_init: float = 0.1,
    ):
        # Hidden layers is the sizes minus input and output layers
        self._n_hidden_layers = len(layer_sizes) - 2
        # Penalization variables
        self._lmbda = lmbda
        self._regularizer = regularizer
        self._regularizer_der = regularizer_der

        # We need to have a separate scheduler for each of the weights/biases,
        # because of dimensionality and saved variables within the scheduler
        # like the momentum.
        self._scheduler_weights = [
            copy.deepcopy(scheduler) for _ in range(self._n_hidden_layers + 1)
        ]
        self._scheduler_biases = [
            copy.deepcopy(scheduler) for _ in range(self._n_hidden_layers + 1)
        ]

        # Error handling
        if self._n_hidden_layers == 0:
            raise ValueError("Must have at least one hidden layer")
        elif (
            isinstance(activation_functions, list)
            and len(activation_functions) != self._n_hidden_layers
        ):
            raise ValueError(
                f"Dimension of activation function list and the number of hidden layers do not match: {len(activation_functions)} vs {self._n_hidden_layers}"
            )

        # Set the cost and its derivative
        self._cost_function = cost
        self._cost_function_grad = (
            jax.grad(self._cost_function, 0) if cost_grad is None else cost_grad
        )

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
            self._activation_functions_der = [
                self._give_activation_func_der(activ_func)
                for activ_func in self._activation_functions
            ]

        # Also set output function and it's derivative
        self._output_function = output_function
        if callable(output_function_der):
            # If the function is callable we assume the output function has been
            # calculated analytically and passed in correct.
            self._output_function_der = output_function_der
        else:
            # If not we have to find it by looking at the output function.
            self._output_function_der = self._give_activation_func_der(output_function)

        # a-s are the values of the layers, while z-s are the values of the layers before activation
        self._h = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]
        self._a = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]

        # Initialize weights/biases
        self._random_seed = random_seed
        np.random.seed(self._random_seed)
        self._weights = []
        self._biases = []
        for i, layer_size in enumerate(layer_sizes[1:]):
            # Since we start enumerating through the list from the first
            # element, i will always be one less than the index of the
            # corresponding layer_size, so i is the index of the previous layer.
            prev_layer_size = layer_sizes[i]
            layer_weights = np.random.randn(prev_layer_size, layer_size)
            layer_biases = np.zeros(layer_size) + biases_init
            self._weights.append(layer_weights)
            self._biases.append(layer_biases)

    def _give_activation_func_der(
        self, activ_func: typing.Callable[[np.ndarray], np.ndarray]
    ) -> typing.Callable[[np.ndarray], np.ndarray]:
        """Takes in a activation function and gives the derivative of it back

        Args:
            activ_func (typing.Callable[[np.ndarray], np.ndarray]): The activation function to find the derivative of.

        Returns:
            typing.Callable[[np.ndarray], np.ndarray]: The derivative of the activation function.
        """
        if activ_func == sigmoid:
            # Use the analytical derivative if we are using sigmoid.
            return sigmoid_der
        elif activ_func == relu:
            # Also use the analytical derivative if we are using relu.
            return relu_der
        elif activ_func == lrelu:
            # Also use the analytical derivative if we are using leaky relu.
            return lrelu_der
        elif activ_func == identity_output_function:
            # Identity function obviously has the constant 1 as derivative.
            return identity_output_function_der
        else:
            # For anything else we can just use automatic differentiation. Here
            # we want this to work on arrays as well so we use vmap.
            return jax.vmap(jax.vmap(jax.grad(activ_func, 0)))

    @property
    def _all_weights(self):
        x = np.array([])
        for w in self._weights:
            x = np.append(x, w.flatten())
        return x

    def _cost(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """The total cost including regularization

        Args:
            y_pred (np.ndarray): The prediction of the network.
            y (np.ndarray): The target.

        Returns:
            float: The cost given the y_pred and y.
        """
        c = self._cost_function(y_pred, y)
        # We check if lambda is not 0 because it is a waste of resources to
        # compute the regularization term if we do not have regularization.
        if self._lmbda != 0:
            c += self._lmbda * self._regularizer(self._all_weights)

        return c

    def _forward_one(self, h: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward to hidden layer `i` (or the output layer if `i` = `number of
        hidden layer plus 1`)
        """
        activation_func = (
            self._activation_functions[i - 1]
            if i <= self._n_hidden_layers
            else self._output_function
        )
        # We have that the weight/bias indices start at 0, so the biases/weights
        # for layer i will have index i-1.
        a = np.array(h @ self._weights[i - 1])
        for j in range(a.shape[0]):
            a[j, :] += self._biases[i - 1]
        h = activation_func(a)

        return a, h

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs forward-propagation on some design matrix X.

        Args:
            X (np.ndarray): The design matrix.

        Returns:
            np.ndarray: The output of the forward propagation algorithm.
        """
        self._h_0 = X
        h_prev = self._h_0
        for i in range(self._n_hidden_layers + 1):
            self._a[i], self._h[i] = self._forward_one(h_prev, i + 1)
            h_prev = self._h[i]

        return self._h[-1]

    def predict(
        self, X: np.ndarray, decision_boundary: float | None = None
    ) -> np.ndarray:
        """Like forward, except includes options for classification.

        Args:
            X (np.ndarray): The design matrix of observations to predict for.
            decision_boundary (float | None): If given, predicts 1 if above the `decision_boundary`, and 0 if below.

        Returns:
            np.ndarray: Predictions of the neural net (categorical or numerical depending on `decision_boundary`)
        """
        y = self.forward(X)
        if decision_boundary is None:
            return y
        else:
            return np.array((y > decision_boundary), dtype=int)

    def _print_performance(self, X: np.ndarray, y: np.ndarray):
        """Print the cost of the current network for some given data.

        Args:
            X (np.ndarray): The design matrix.
            y (np.ndarray): The reponse vector.
        """
        y_pred = self.forward(X)
        print(f"COST: {self._cost_history[-1]}")

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

        g = self._cost_function_grad(y_pred, y)
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
            weight_grads[k - 1] = h_k_1.T @ g + self._lmbda * self._regularizer_der(W_k)
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
        minibatch_size=10,
        print_every: int = 0,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        seed: int = None,
    ):
        """Train the neural network to some data

        Args:
            X (np.array): The design matrix.
            y (np.array): The response. Must be of compatible dimensions as X.
            epochs (int, optional): The max amount of epochs to allow. Defaults to 100.
            sgd (bool, optional): Wether to use stochastic gradient descent or not. Defaults to False.
            minibatch_size (int, optional): The size of each mini-batch when using stochastic gradient descent. Defaults to 10.
            print_every (int, optional): Print the cost every `print_every` epoch. If 0 do not print at all. Defaults to 0.
            tol (float, optional): Determines convergence if the maximum difference over the last `n_iter_no_change` is less than this. Defaults to 1e-4.
            n_iter_no_change (int, optional): The amount of iterations back we look for convergence over. Default to 10.
        """
        # Set the seed again
        np.random.seed(seed if not seed is None else self._random_seed)
        # Reset all the schedulers
        for k in range(self._n_hidden_layers + 1):
            self._scheduler_weights[k].reset()
            self._scheduler_biases[k].reset()

        # Fail if the sgd size is too big for the data
        if sgd and X.shape[0] < minibatch_size:
            raise ValueError(
                f"The size of a minibatch in stochastic gradient descent cannot be bigger than the amount of datapoints (datapoints: {X.shape[0]}, minibatch_size: {minibatch_size})"
            )

        self._cost_history = []
        for i in range(epochs):
            y_pred = self.forward(X)
            current_cost = self._cost(y_pred, y)
            max_diff = (
                None
                if len(self._cost_history) < n_iter_no_change
                else max(
                    (
                        abs(current_cost - cost)
                        for cost in self._cost_history[-n_iter_no_change:]
                    )
                )
            )
            if max_diff is not None and max_diff < tol:
                print(f"Convergence after {i} epochs")
                break
            self._cost_history.append(current_cost)

            n_iter_per_epoch = 1 if not sgd else int(X.shape[0] / minibatch_size)
            for _ in range(n_iter_per_epoch):
                X_data = X
                y_data = y
                # Pick random datapoints of size given by minibatch_size if we
                # want to use minibatches to train on.
                if sgd:
                    rand_indices = np.random.choice(
                        X.shape[0], minibatch_size, replace=False
                    )
                    X_data = X[rand_indices, :]
                    y_data = y[rand_indices]

                self._backward_once(X_data, y_data)

            if print_every > 0 and i % print_every == 0:
                self._print_performance(X, y)
