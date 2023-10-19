import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable
import abc


class Optimizer(abc.ABC):
    def __init__(
        self,
        X,
        y,
        cost: Callable[[np.array, np.array, np.array], float] | None = None,
        cost_grad: Callable[[np.array, np.array, np.array], np.array] | str = "jax",
        use_momentum=False,
        momentum_alpha=0.9,
        timedecay=lambda t: 1,
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

        self.use_momentum = use_momentum
        self.momentum_alpha = momentum_alpha
        self.timedecay = timedecay
        self.random_seed = random_seed

    def cost_grad_jax(self, cost):
        return jax.grad(cost, 2)

    @property
    def learning_rate(self):
        return self.timedecay(self._current_epoch) * self._learning_rate

    @abc.abstractmethod
    def _theta_update(self, grad) -> np.array:
        pass

    def momentum(self, v):
        return self.momentum_alpha * v

    def run_on_data(self, data_X, data_y):
        grad = self.cost_grad(data_X, data_y, self._theta)

        new_update = self._theta_update(grad)

        if self.use_momentum:
            new_update += self.momentum(self._update)

        self._update = new_update
        self._theta += self._update
        self._current_epoch += 1

    def _set_vars(self, learning_rate):
        """
        Sets variables before training
        """
        np.random.seed(self.random_seed)
        self._theta = np.random.randn(self.X.shape[1], 1).flatten()
        self._update = self.np.zeros_like(self._theta)
        self._learning_rate = learning_rate
        self._current_epoch = 0

    def optimize(self, epochs: int, learning_rate: float) -> np.array:
        """
        Optimize on the whole dataset, using gradient descent or some other method.
        """
        self._set_vars(learning_rate)
        for _ in range(epochs):
            self.run_on_data(self.X, self.y)

        return self._theta

    def optimize_stochastic(
        self, epochs: int, learning_rate: float, sgd_size
    ) -> np.array:
        """
        Stochastic version of optimize. We do this by calculating the gradient
        only on stochastic mini-batches.
        """
        np.random.seed(self.random_seed)
        self._set_vars(learning_rate)
        n_batches = int(self.X.shape[0] / sgd_size)
        for _ in range(epochs):
            for _ in range(n_batches):
                # Just choose some random indices
                rand_indices = np.random.choice(
                    self.X.shape[0], sgd_size, replace=False
                )
                X_sgd = self.X[rand_indices, :]
                y_sgd = self.y[rand_indices]
                self.run_on_data(X_sgd, y_sgd)

        return self._theta


class OptimizerGD(Optimizer):
    def _theta_update(self, grad):
        return -self.learning_rate * grad


class OptimizerAdagrad(Optimizer):
    def _theta_update(self, grad, delta=1e-7):
        self.r += grad**2
        return -self.learning_rate / (delta + self.np.sqrt(self.r)) * grad

    def _set_vars(self, learning_rate):
        super()._set_vars(learning_rate)
        self.r = self.np.zeros_like(self._theta)


class OptimizerRMSProp(Optimizer):
    def __init__(
        self,
        X,
        y,
        cost: Callable[[np.array, np.array, np.array], float] | None = None,
        cost_grad: Callable[[np.array, np.array, np.array], np.array] | str = "jax",
        rmsprop_decay=0.99,
    ):
        super().__init__(X, y, cost, cost_grad)
        self._rmsprop_decay = rmsprop_decay

    def _theta_update(self, grad, delta=1e-6):
        self.r = self._rmsprop_decay * self.r + (1 - self._rmsprop_decay) * grad**2
        return -self.learning_rate / (self.np.sqrt(delta + self.r)) * grad

    def _set_vars(self, learning_rate):
        super()._set_vars(learning_rate)
        self.r = self.np.zeros_like(self._theta)


class OptimizerAdam(Optimizer):
    def __init__(
        self,
        X,
        y,
        cost: Callable[[np.array, np.array, np.array], float] | None = None,
        cost_grad: Callable[[np.array, np.array, np.array], np.array] | str = "jax",
        adam_decay1=0.9,
        adam_decay2=0.999,
    ):
        super().__init__(X, y, cost, cost_grad)
        self.adam_decay1 = adam_decay1
        self.adam_decay2 = adam_decay2

    def _theta_update(self, grad, delta=1e-6):
        self.t += 1
        self.s = self.adam_decay1 * self.s + (1 - self.adam_decay1) * grad
        self.r = self.adam_decay2 * self.r + (1 - self.adam_decay2) * grad**2
        s_hat = self.s / (1 - self.adam_decay1**self.t)
        r_hat = self.r / (1 - self.adam_decay2**self.t)
        return -self.learning_rate * s_hat / (self.np.sqrt(r_hat) + delta)

    def _set_vars(self, learning_rate):
        super()._set_vars(learning_rate)
        self.r = self.np.zeros_like(self._theta)
        self.s = self.np.zeros_like(self._theta)
        self.t = 0
