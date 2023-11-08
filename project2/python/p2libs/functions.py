#
# A file containing some useful functions useful for various parts of our
# project, like cost-function, activation-functions and more.
#
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x: np.ndarray) -> np.ndarray:
    """
    The derivative of the sigmoid activation function.
    """
    sigmoid_val = sigmoid(x)
    return sigmoid_val * (1 - sigmoid_val)


def relu(x: np.ndarray) -> np.ndarray:
    """
    The relu activation function.
    """
    return np.where(x > np.zeros_like(x), x, np.zeros_like(x))


def relu_der(x: np.ndarray) -> np.ndarray:
    """
    The derivative of the relu activation function.
    """
    return np.where(x > np.zeros_like(x), 1, 0)


def lrelu(x: np.ndarray, delta=1e-3) -> np.ndarray:
    """
    The leaky relu activation function.
    """
    return np.where(x > np.zeros_like(x), x, delta * x)


def lrelu_der(x: np.ndarray, delta=1e-3):
    """
    The derivative of the leaky relu activation function.
    """
    return np.where(x > 0, 1, delta)


def sse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    The total sum of squares. May be used as a cost-function.
    """
    diff = y - y_pred
    return (diff.T @ diff)[0, 0]


def sse_grad(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    The derivative of the sum of squares with respect to the predictions. This
    is needed for use in a neural network.
    """
    return 2 * (y_pred - y)


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    The mean squared error. May be used as a cost-function.
    """
    return sse(y_pred, y) / len(y)


def cross_entropy(y_pred: np.ndarray, y: np.ndarray, delta=1e-10) -> float:
    """
    The cross-entropy. Minimizing this is the equivalent of maximizing the
    likelihood. May be used as a cost-function when we have binary
    classification.
    """
    return -np.sum(y * np.log(y_pred + delta) + (1 - y) * np.log(1 - y_pred + delta))


def cross_entropy_grad(y_pred: np.ndarray, y: np.ndarray, delta=1e-10) -> np.ndarray:
    # When using relu/lrelu or other activation functions with the sigmoid some
    # of the predictions can initially become very close to 0/1 in which case we
    # could end up dividing by 0 in practice. To avoid this we simply clip the
    # predictions to not become too close to 0/1.
    y_pred_clipped = np.clip(y_pred, delta, 1 - delta)
    return (y_pred_clipped - y) / ((1 - y_pred_clipped) * y_pred_clipped)


def identity_output_function(x: np.ndarray) -> np.ndarray:
    """
    A simple identity function. Useful as an output function whenwe want
    regression.
    """
    return x


def identity_output_function_der(x: np.ndarray) -> np.ndarray:
    """
    A trivially simple derivative of the identity function, which works on numpy arrays
    """
    return np.ones_like(x)


def l2regularizer(theta: np.ndarray) -> np.ndarray:
    return theta
