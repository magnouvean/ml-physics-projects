import abc
import typing

import numpy as np


class Scheduler(abc.ABC):
    def __init__(
        self,
        learning_rate,
        timedecay: typing.Callable[[int], float] = lambda _: 1,
        use_momentum=False,
        momentum_alpha=0.9,
    ):
        self._learning_rate = learning_rate
        self._momentum_alpha = momentum_alpha
        self._timedecay = timedecay
        self._update: np.ndarray | None = None
        self._use_momentum = use_momentum
        self.reset()

    @property
    def learning_rate(self):
        return self._learning_rate * self._timedecay(self._current_epoch)

    @abc.abstractmethod
    def _update_no_momentum(self, grad):
        pass

    def update(self, grad) -> np.ndarray:
        self._current_epoch += 1
        if self._use_momentum:
            if self._update is None:
                self._update = np.zeros_like(grad)

            self._update = self._update_no_momentum(grad) + self.momentum()
        else:
            self._update = self._update_no_momentum(grad)

        return self._update

    def momentum(self):
        return self._momentum_alpha * self._update

    def reset(self):
        self._current_epoch = 0


class SchedulerConstant(Scheduler):
    def _update_no_momentum(self, grad):
        return -self.learning_rate * grad


class SchedulerAdagrad(Scheduler):
    def __init__(
        self,
        learning_rate,
        timedecay: typing.Callable[[int], float] = lambda _: 1,
        use_momentum=False,
        momentum_alpha=0.9,
    ):
        super().__init__(learning_rate, use_momentum=use_momentum)

    def _update_no_momentum(self, grad, delta=1e-7):
        self.r += grad**2
        return -self.learning_rate / (delta + np.sqrt(self.r)) * grad

    def reset(self):
        super().reset()
        self.r = 0


class SchedulerRMSProp(Scheduler):
    def __init__(
        self,
        learning_rate,
        timedecay: typing.Callable[[int], float] = lambda _: 1,
        rmsprop_decay=0.99,
    ):
        super().__init__(learning_rate, timedecay)
        self._rmsprop_decay = rmsprop_decay

    def _update_no_momentum(self, grad, delta=1e-6):
        self._r = self._rmsprop_decay * self._r + (1 - self._rmsprop_decay) * grad**2
        return -self.learning_rate / (np.sqrt(delta + self._r)) * grad

    def reset(self):
        super().reset()
        self._r = 0


class SchedulerAdam(Scheduler):
    def __init__(
        self,
        learning_rate,
        timedecay: typing.Callable[[int], float] = lambda _: 1,
        adam_decay1=0.9,
        adam_decay2=0.999,
    ):
        super().__init__(learning_rate, timedecay)
        self._adam_decay1 = adam_decay1
        self._adam_decay2 = adam_decay2

    def _update_no_momentum(self, grad, delta=1e-6):
        self._t += 1
        self._s = self._adam_decay1 * self._s + (1 - self._adam_decay1) * grad
        self._r = self._adam_decay2 * self._r + (1 - self._adam_decay2) * grad**2
        s_hat = self._s / (1 - self._adam_decay1**self._t)
        r_hat = self._r / (1 - self._adam_decay2**self._t)
        return -self.learning_rate * s_hat / (np.sqrt(r_hat) + delta)

    def reset(self):
        super().reset()
        self._r = 0
        self._s = 0
        self._t = 0
