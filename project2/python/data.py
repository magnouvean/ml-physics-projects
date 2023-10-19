import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt

np.random.seed(1234)

f = lambda x: 4 - 6 * x**2 + x**4
x = np.linspace(-2, 2, 1001)
X = np.c_[np.ones(len(x)), x, x**2, x**3, x**4]
y = f(x)
plt.plot(x, f(x))

analytical = np.linalg.pinv(X.T @ X) @ X.T @ y


def cost_mse_ols(X, y, theta):
    a = y - X @ theta
    return jnp.sum(a**2)
