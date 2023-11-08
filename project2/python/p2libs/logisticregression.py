import typing

import numpy as np

from .functions import l2regularizer
from .neuralnet import sigmoid
from .optimizer import Optimizer
from .schedulers import Scheduler


def logistic_regression_cost_grad(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    y_hat = sigmoid(X @ beta)  # .reshape((len(y), 1))
    return -X.T @ (y - y_hat)


class LogisticRegression:
    def __init__(
        self,
        scheduler: Scheduler,
        lmbda: float = 0,
        regularizer: typing.Callable[[np.ndarray], np.ndarray] = l2regularizer,
    ):
        """Constructor for the logistic regression model.

        Args:
            scheduler (Scheduler): _description_
            regularizer (typing.Callable[[np.ndarray], np.ndarray]): The
            derivative of the regularization term (without the lambda) that is
            included in the regularized cost-function. Defaults to l2regularizer.
            lmbda (float, optional): The regularization parameter. Defaults to 0
            (no regularization).
        """

        self._scheduler = scheduler
        self._lmbda = lmbda
        self._regularizer = l2regularizer

    def _cost_grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return logistic_regression_cost_grad(
            X, y, beta
        ) + self._lmbda * self._regularizer(beta)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100):
        optimizer = Optimizer(X, y, self._scheduler, cost_grad=self._cost_grad)
        self._betas = optimizer.optimize(epochs)

    def predict(
        self, X: np.ndarray, probs=True, decision_boundary: float = 0.5
    ) -> np.ndarray:
        p = sigmoid(X @ self._betas)
        return p if probs else np.array(p > decision_boundary, dtype=int)
