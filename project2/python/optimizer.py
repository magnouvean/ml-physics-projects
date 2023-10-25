import abc
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from schedulers import Scheduler


class Optimizer(abc.ABC):
    def __init__(
        self,
        X,
        y,
        scheduler: Scheduler,
        cost: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        cost_grad: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        | str = "jax",
        random_seed: int = 1234,
    ):
        self.X = X
        self.y = y
        self.np = np if cost_grad != "jax" else jnp
        if callable(cost_grad):
            self.cost_grad = cost_grad
        elif cost_grad == "jax":
            if not cost:
                raise ValueError(
                    "Given jax as method for determining the gradient of the cost, but no cost was given"
                )
            self.cost_grad = self.cost_grad_jax(cost)
        else:
            raise ValueError(f"Unsupported value for cost_gradient: {cost_grad}")

        self._scheduler = scheduler
        self.random_seed = random_seed

    def cost_grad_jax(self, cost):
        return jax.grad(cost, 2)

    def _run_on_data(self, data_X, data_y):
        grad = self.cost_grad(data_X, data_y, self._theta)

        theta_update = self._scheduler.update(grad)
        self._theta += theta_update

    def _set_vars(self):
        """
        Sets variables before training
        """
        np.random.seed(self.random_seed)
        self._theta = np.random.randn(self.X.shape[1], 1).flatten()
        self._scheduler.reset()

    def optimize(self, epochs: int) -> np.ndarray:
        """
        Optimize on the whole dataset, using gradient descent or some other method.
        """
        self._set_vars()
        for _ in range(epochs):
            self._run_on_data(self.X, self.y)

        return self._theta

    def optimize_stochastic(self, epochs: int, sgd_size: int) -> np.ndarray:
        """
        Stochastic version of optimize. We do this by calculating the gradient
        only on stochastic mini-batches.
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

        return self._theta
