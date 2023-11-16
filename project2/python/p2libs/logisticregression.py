import typing

import numpy as np

from .functions import l2regularizer_der, sigmoid
from .optimizer import Optimizer
from .schedulers import Scheduler


def logistic_regression_cost(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    X_beta = X @ beta
    return -np.sum(
        [y[i] * X_beta[i] - np.log(1 + np.exp(X_beta)) for i in range(X.shape[0])]
    )


def logistic_regression_cost_grad(
    X: np.ndarray, y: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    y_pred = sigmoid(X @ beta)
    return -X.T @ (y - y_pred)


class LogisticRegression:
    def __init__(
        self,
        scheduler: Scheduler,
        lmbda: float = 0,
        regularizer_der: typing.Callable[[np.ndarray], np.ndarray] = l2regularizer_der,
    ):
        """Constructor for the logistic regression model.

        Args:
            scheduler (Scheduler): _description_
            regularizer_der (typing.Callable[[np.ndarray], np.ndarray]): The
            derivative of the regularization term (without the lambda) that is
            included in the regularized cost-function. Defaults to
            l2regularizer_der.
            lmbda (float, optional): The regularization parameter. Defaults to 0
            (no regularization).
        """

        self._scheduler = scheduler
        self._lmbda = lmbda
        self._regularizer_der = regularizer_der

    def _cost_grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        y_pred = sigmoid(X @ beta)
        return logistic_regression_cost_grad(
            X, y, beta
        ) + self._lmbda * self._regularizer_der(beta)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=500):
        self._optimizer = Optimizer(
            X,
            y,
            self._scheduler,
            cost=logistic_regression_cost,
            cost_grad=self._cost_grad,
            n_iters_no_change=10,
            tol=1e-4,
        )
        self._betas = self._optimizer.optimize(epochs)

    def predict(
        self, X: np.ndarray, probs=True, decision_boundary: float = 0.5
    ) -> np.ndarray:
        """Make a prediction for some design matrix X.

        Args:
            X (np.ndarray): The design matrix to make predictions for.
            probs (bool): If true give out the results as a probability, and if
            not as a classification determined by decision_boundary. Defaults
            to True.
            decision_boundary (float): Classify to 1 if above this boundary,
            and 0 if below. Only relevant when probs=False. Defaults to 0.5

        """
        p = sigmoid(X @ self._betas)
        return p if probs else np.array(p > decision_boundary, dtype=int)
