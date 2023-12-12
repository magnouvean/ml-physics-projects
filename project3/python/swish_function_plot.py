import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 101)


def swish(x):
    return x / (1 + np.exp(-x))


plt.plot(x, swish(x))
plt.xlabel("x")
plt.ylabel("swish(x)")
plt.savefig("../figures/swish_plot.png", dpi=196)
