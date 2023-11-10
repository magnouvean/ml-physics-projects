import abc
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .schedulers import Scheduler


class Optimizer(abc.ABC):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scheduler: Scheduler,
        cost: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        cost_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        | str = "jax",
        random_seed: int = 1234,
        tol: float = 1e-4,
        n_iters_no_change: int = None,
    ):
        self.X = X
        self.y = y
        self.np = np if cost_grad != "jax" else jnp
        if callable(cost_grad):
            self.cost_grad = cost_grad
        elif cost_grad == "jax":
            self.cost_grad = self.cost_grad_jax(cost)
        else:
            raise ValueError(f"Unsupported value for cost_gradient: {cost_grad}")

        self._scheduler = scheduler
        self.random_seed = random_seed
        self._tol = tol
        self._n_iters_no_change = n_iters_no_change

    def cost_grad_jax(self, cost):
        return jax.grad(cost, 2)

    def _run_on_data(self, data_X: np.ndarray, data_y: np.ndarray) -> bool:
        """Run one iteration of newtons method on some data X (design matrix) and y (response).

        Args:
            data_X (np.ndarray): The design matrix.
            data_y (np.ndarray): The response/target.

        Returns:
            bool: Indicator which tells us if we have achieved convergence.
        """
        grad = self.cost_grad(data_X, data_y, self._theta)

        theta_update = self._scheduler.update(grad)
        self._theta += theta_update

    def _has_converged(self) -> bool:
        """Determines whether we have convergence or not.

        Returns:
            bool: Indicator for whether we have converged.
        """
        # If _n_iters_no_change is None, we do not check for convergence.
        if self._n_iters_no_change is None:
            return False

        # _n_epoch is just a counter variable for how many times we have
        # checked convergence, which in our case increases by 1 per epoch. This
        # is only used for printing out the amount of epochs it took before
        # convergence.
        self._n_epoch += 1
        current_cost = self._cost(self._X, self._y, self._theta)
        max_diff = (
            None
            if len(self._cost_history) < self._n_iters_no_change
            else max(
                [
                    abs(current_cost - c)
                    for c in self._cost_history[-self._n_iters_no_change :]
                ]
            )
        )
        self._cost_history.append(current_cost)
        converged = max_diff is not None and max_diff < self._tol
        if converged:
            print(f"Convergence after {self._n_epoch} epochs")
        return converged

    def _set_vars(self):
        """Sets/resets some variables before training"""
        np.random.seed(self.random_seed)
        self._theta = np.random.randn(self.X.shape[1], 1).flatten()
        self._scheduler.reset()
        self._cost_history = []
        self._n_epoch = 0

    def optimize(self, epochs: int) -> np.ndarray:
        """Optimize on the whole dataset, without stochastic mini-batches.

        Args:
            epochs (int): The max amount of epochs to allow.

        Returns:
            np.ndarray: The parameter theta which minimizes the cost.
        """
        self._set_vars()
        for _ in range(epochs):
            self._run_on_data(self.X, self.y)

            if self._has_converged():
                break

        return self._theta

    def optimize_stochastic(self, epochs: int, sgd_size: int) -> np.ndarray:
        """Stochastic version of optimize. Functions exactly the same, but
        draws some amount of mini-batches for each epoch.

        Args:
            epochs (int): The max amount of epochs to allow.
            sgd_size (int): The size of each mini-batch.

        Returns:
            np.ndarray: The parameter theta which minimizes the cost.
        """
        np.random.seed(self.random_seed)
        self._set_vars()
        n_batches = int(self.X.shape[0] / sgd_size)
        for _ in range(epochs):
            for _ in range(n_batches):
                # Just choose some random indices
                rand_indices = np.random.choice(
                    self.X.shape[0], sgd_size, replace=False
                )
                X_sgd = self.X[rand_indices, :]
                y_sgd = self.y[rand_indices]
                self._run_on_data(X_sgd, y_sgd)

            if self._has_converged():
                break

        return self._theta
