import numpy as np

from neuralnet import sigmoid
from optimizer import Optimizer
from schedulers import Scheduler


def logistic_regression_cost_der(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    y_hat = sigmoid(X @ theta)
    return -X.T @ (y - y_hat)


class LogisticRegression:
    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100) -> np.ndarray:
        optimizer = Optimizer(
            X, y, self._scheduler, cost_grad=logistic_regression_cost_der
        )
        self._betas = optimizer.optimize(epochs)

    def predict(
        self, X: np.ndarray, probs=True, decision_boundary: float = 0.5
    ) -> np.ndarray:
        p = sigmoid(X @ self._betas)
        return p if probs else p > decision_boundary
