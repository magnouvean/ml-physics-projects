import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from p2libs import (
    Optimizer,
    SchedulerAdagrad,
    SchedulerAdam,
    SchedulerConstant,
    SchedulerRMSProp,
    fig_directory,
    mse,
)

np.random.seed(1234)


def f(x):
    """A basic forth order polynomial to generate our response from."""
    return 4 - 6 * x**2 + x**4


x = np.linspace(-2, 2, 201)
X = np.c_[np.ones(len(x)), x, x**2, x**3, x**4]
y = f(x)

# Analytical least squares solution.
analytical = np.linalg.pinv(X.T @ X) @ X.T @ y
print(f"Analytical solution: {analytical}")


# Calculate some sensible learning_rate for gradient descent.
hesse = 2 * X.T @ X
eigvalues, _ = np.linalg.eig(hesse)
opt_learning_rate = 1.0 / np.max(eigvalues)


# We also define a decay function to see if this improves the speed of
# convergence.
class DecayFunctionExample:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def __call__(self, t):
        return self.t0 / (t + self.t1)


def optimize_demo(
    cost,
    cost_grad,
    timedecay_func,
    learning_rates_plain,
    learning_rates_plain_sgd,
    learning_rates_adagrad,
    epochs,
    minibatch_size,
) -> tuple[
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """Run optimization of some parameter for a cost function on the data generated.

    Returns:
        Two tuples each of size 8, with the first one being the costs for each
        of the methods and the second being the estimated parameters. The order
        is gd, sgd, gd with momentum, sgd with momentum, adagrad gd, adagrad
        sgd, adagrad gd with momentum, adagrad sgd with momentum.
    """
    # Define some learning_rates these are just gathered from what appeared to
    # give good results from trial and error. One could perhaps try to tweak
    # these further.
    betas_gd = np.zeros((len(learning_rates_plain), X.shape[1]))
    betas_sgd = np.zeros_like(betas_gd)
    betas_gd_mom = np.zeros_like(betas_gd)
    betas_sgd_mom = np.zeros_like(betas_gd)
    betas_adagrad_gd = np.zeros_like(betas_gd)
    betas_adagrad_sgd = np.zeros_like(betas_gd)
    betas_adagrad_gd_mom = np.zeros_like(betas_gd)
    betas_adagrad_sgd_mom = np.zeros_like(betas_gd)
    costs_gd = np.zeros_like(learning_rates_plain)
    costs_sgd = np.zeros_like(costs_gd)
    costs_gd_mom = np.zeros_like(costs_gd)
    costs_sgd_mom = np.zeros_like(costs_gd)
    costs_adagrad_gd = np.zeros_like(costs_gd)
    costs_adagrad_sgd = np.zeros_like(costs_gd)
    costs_adagrad_gd_mom = np.zeros_like(costs_gd)
    costs_adagrad_sgd_mom = np.zeros_like(costs_gd)

    for i, (
        learning_rate_plain,
        learning_rate_plain_sgd,
        learning_rate_adagrad,
    ) in enumerate(
        zip(learning_rates_plain, learning_rates_plain_sgd, learning_rates_adagrad)
    ):
        # Plain gradient descent.
        scheduler_gd = SchedulerConstant(learning_rate_plain)
        scheduler_sgd = SchedulerConstant(learning_rate_plain_sgd, timedecay_func)
        o_gd = Optimizer(X, y, scheduler_gd, cost, cost_grad)
        o_sgd = Optimizer(X, y, scheduler_sgd, cost, cost_grad)
        betas_gd[i, :] = o_gd.optimize(epochs)
        betas_sgd[i, :] = o_sgd.optimize_stochastic(epochs, minibatch_size)

        # With momentum (both stochastic and non-stochastic).
        scheduler_gd_mom = SchedulerConstant(learning_rate_plain, use_momentum=True)
        scheduler_sgd_mom = SchedulerConstant(
            learning_rate_plain_sgd, timedecay_func, use_momentum=True
        )
        o_gd_mom = Optimizer(X, y, scheduler_gd_mom, cost, cost_grad)
        o_sgd_mom = Optimizer(X, y, scheduler_sgd_mom, cost, cost_grad)
        betas_gd_mom[i, :] = o_gd_mom.optimize(epochs)
        betas_sgd_mom[i, :] = o_sgd_mom.optimize_stochastic(epochs, minibatch_size)

        # And lastly adagrad.
        scheduler_adagrad = SchedulerAdagrad(learning_rate_adagrad)
        scheduler_adagrad_mom = SchedulerAdagrad(
            learning_rate_adagrad, use_momentum=True
        )
        o_adagrad = Optimizer(X, y, scheduler_adagrad, cost, cost_grad)
        o_adagrad_mom = Optimizer(X, y, scheduler_adagrad_mom, cost, cost_grad)
        betas_adagrad_gd[i, :] = o_adagrad.optimize(epochs)
        betas_adagrad_sgd[i, :] = o_adagrad.optimize_stochastic(epochs, minibatch_size)
        betas_adagrad_gd_mom[i, :] = o_adagrad_mom.optimize(epochs)
        betas_adagrad_sgd_mom[i, :] = o_adagrad_mom.optimize_stochastic(
            epochs, minibatch_size
        )

        costs_gd[i] = cost(X, y, betas_gd[i, :])
        costs_sgd[i] = cost(X, y, betas_sgd[i, :])
        costs_gd_mom[i] = cost(X, y, betas_gd_mom[i, :])
        costs_sgd_mom[i] = cost(X, y, betas_sgd_mom[i, :])
        costs_adagrad_gd[i] = cost(X, y, betas_adagrad_gd[i, :])
        costs_adagrad_sgd[i] = cost(X, y, betas_adagrad_sgd[i, :])
        costs_adagrad_gd_mom[i] = cost(X, y, betas_adagrad_gd_mom[i, :])
        costs_adagrad_sgd_mom[i] = cost(X, y, betas_adagrad_sgd_mom[i, :])

    return (
        (
            costs_gd,
            costs_sgd,
            costs_gd_mom,
            costs_sgd_mom,
            costs_adagrad_gd,
            costs_adagrad_sgd,
            costs_adagrad_gd_mom,
            costs_adagrad_sgd_mom,
        ),
        (
            betas_gd,
            betas_sgd,
            betas_gd_mom,
            betas_sgd_mom,
            betas_adagrad_gd,
            betas_adagrad_sgd,
            betas_adagrad_gd_mom,
            betas_adagrad_sgd_mom,
        ),
    )


# We will here first optimize the MSE, and then move on to ridge afterwards.
def mse_ols(X, y, beta):
    y_pred = X @ beta
    return np.mean((y - y_pred) ** 2)


def mse_ols_grad(X, y, beta):
    return 2 * X.T @ (X @ beta - y) / X.shape[0]


# Make grids of learning rates we test. Here we have constructed these so that
# the last cost will diverge, while the other will converge.
learning_rates_plain = 10 ** (np.linspace(-5, -1, 11))
learning_rates_plain_sgd = 10 ** (np.linspace(-5, 0, 11))
learning_rates_adagrad = 10 ** (np.linspace(-5, 5, 11))

# And we create an example decay function out of this.
somedecayfunc = DecayFunctionExample(1.0, 10)
# We keep the amount of epochs and minibatch_size fixed when we compare the
# different learning-rates/schedulers.
epochs = 100
minibatch_size = 16

(
    costs_gd,
    costs_sgd,
    costs_gd_mom,
    costs_sgd_mom,
    costs_adagrad_gd,
    costs_adagrad_sgd,
    costs_adagrad_gd_mom,
    costs_adagrad_sgd_mom,
), _ = optimize_demo(
    mse_ols,
    mse_ols_grad,
    somedecayfunc,
    learning_rates_plain,
    learning_rates_plain_sgd,
    learning_rates_adagrad,
    epochs,
    minibatch_size,
)

# Generate figures for gd/sgd. Keep in mind that since we have created these so
# that we ignore the last cost (in the case of gd the last 2 costs) as these
# have diverged.
fig, ax = plt.subplots(2, 2)
for i, (title, cost, learning_rates) in enumerate(
    (
        ("GD", costs_gd[:-2], learning_rates_plain[:-2]),
        ("SGD", costs_sgd[:-1], learning_rates_plain_sgd[:-1]),
        ("GD (With momentum)", costs_gd_mom[:-1], learning_rates_plain[:-1]),
        ("SGD (With momentum)", costs_sgd_mom[:-1], learning_rates_plain_sgd[:-1]),
    )
):
    r = i // 2
    c = i % 2
    ax[r, c].plot(np.log(learning_rates), cost)
    ax[r, c].set_title(title)
    ax[r, c].set_xlabel("log($\\eta$)")
    ax[r, c].set_ylabel("MSE")

fig.subplots_adjust(hspace=0.6, wspace=0.3, bottom=0.1)
fig.savefig(f"{fig_directory}/optimizers_plot.png", dpi=196)


#
# We see what happens when we give it four times as many epochs
#
more_epochs = epochs * 4
print("\n\n====Increaing amount of epochs====")
scheduler = SchedulerConstant(0.01)
o = Optimizer(X, y, scheduler, mse_ols, mse_ols_grad)

beta_gd = o.optimize(epochs)
beta_gd_more = o.optimize(more_epochs)
beta_sgd = o.optimize_stochastic(epochs, minibatch_size)
beta_sgd_more = o.optimize_stochastic(more_epochs, minibatch_size)

cost_gd = mse_ols(X, y, beta_gd)
cost_gd_more = mse_ols(X, y, beta_gd_more)
cost_sgd = mse_ols(X, y, beta_sgd)
cost_sgd_more = mse_ols(X, y, beta_sgd_more)

print(f"COST GD {epochs} epochs: {cost_gd}, {more_epochs} epochs: {cost_gd_more}")
print(f"COST SGD {epochs} epochs: {cost_sgd}, {more_epochs} epochs: {cost_sgd_more}")


#
# We may also do a quick check to see the effect minibatch_size has
#
print("\n\n====Varying the SGD size====")
for minibatch_size in (8, 12, 16, 32):
    costs = np.zeros_like(learning_rates_plain_sgd)
    for i, learning_rate in enumerate(learning_rates_plain_sgd):
        scheduler = SchedulerConstant(learning_rate)
        o = Optimizer(X, y, scheduler, mse_ols, mse_ols_grad)
        beta = o.optimize_stochastic(epochs, minibatch_size=minibatch_size)
        costs[i] = mse_ols(X, y, beta)

    print(f"MINIBATCH SIZE {minibatch_size}, best cost: {np.nanmin(costs)}")


#
# We now illustrate that we can also use jax
#
costs, betas = optimize_demo(
    mse_ols,
    "jax",
    somedecayfunc,
    learning_rates_plain,
    learning_rates_plain_sgd,
    learning_rates_adagrad,
    epochs,
    minibatch_size,
)

# And we also then can gather the best beta for each of the different
# schedulers.
for method_name, cost, beta in zip(
    (
        "GD",
        "SGD",
        "GD with momentum",
        "SGD with momentum",
        "Adagrad GD",
        "Adagrad SGD",
        "Adagrad GD with momentum",
        "Adagrad SGD with momentum",
    ),
    costs,
    betas,
):
    print(f"\n\n===={method_name}====")
    best_cost_index = np.nanargmin(cost)
    print(f"Best cost: {cost[best_cost_index]}")
    print(f"Best beta: {beta[best_cost_index,:]}")


# We use a class which lets us save the regularization parameter within
# functions.
class RidgeCost:
    def __init__(self, lmbda: float):
        self._lmbda = lmbda

    def cost(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        return mse_ols(X, y, beta) + self._lmbda * np.sum(beta**2)

    def grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return mse_ols_grad(X, y, beta) + self._lmbda * beta


regularization_parameters = 10 ** (np.linspace(-8, 1, 10))
# Collect a matrix of test mses which in (i, j) has the mse of the test data
# with learning-rate i, regularization parameter j.
test_mses_gd = np.zeros((len(learning_rates_plain), len(regularization_parameters)))
test_mses_sgd = np.zeros(
    (len(learning_rates_plain_sgd), len(regularization_parameters))
)
test_mses_gd_mom = np.zeros_like(test_mses_gd)
test_mses_sgd_mom = np.zeros_like(test_mses_sgd)

test_mses_adagrad_gd = np.zeros(
    (len(learning_rates_adagrad), len(regularization_parameters))
)
test_mses_adagrad_sgd = np.zeros_like(test_mses_adagrad_gd)
test_mses_adagrad_gd_mom = np.zeros_like(test_mses_adagrad_gd)
test_mses_adagrad_sgd_mom = np.zeros_like(test_mses_adagrad_gd)

for j, lmbda in enumerate(regularization_parameters):
    ridge_cost = RidgeCost(lmbda)
    cost = ridge_cost.cost
    cost_grad = ridge_cost.grad

    # We'll here generate some test data as well
    np.random.seed(1234)
    x_test = np.random.uniform(-2, 2, 51)
    X_test = np.c_[np.ones(len(x_test)), x_test, x_test**2, x_test**3, x_test**4]
    y_test = f(x_test)

    _, betas = optimize_demo(
        cost,
        cost_grad,
        somedecayfunc,
        learning_rates_plain,
        learning_rates_plain_sgd,
        learning_rates_adagrad,
        epochs,
        minibatch_size,
    )
    # We here use the fact that all the betas have the same shape (because all
    # the learning-rate grids are of the same size).
    for i in range(betas[0].shape[0]):
        test_mses_gd[i, j] = mse_ols(X_test, y_test, betas[0][i, :])
        test_mses_sgd[i, j] = mse_ols(X_test, y_test, betas[1][i, :])
        test_mses_gd_mom[i, j] = mse_ols(X_test, y_test, betas[2][i, :])
        test_mses_sgd_mom[i, j] = mse_ols(X_test, y_test, betas[3][i, :])
        test_mses_adagrad_gd[i, j] = mse_ols(X_test, y_test, betas[4][i, :])
        test_mses_adagrad_sgd[i, j] = mse_ols(X_test, y_test, betas[5][i, :])
        test_mses_adagrad_gd_mom[i, j] = mse_ols(X_test, y_test, betas[6][i, :])
        test_mses_adagrad_sgd_mom[i, j] = mse_ols(X_test, y_test, betas[7][i, :])


# We now create heat maps from this test loss.
fig, ax = plt.subplots(2, 2)
for i, (title, mse, learning_rates) in enumerate(
    (
        ("GD", test_mses_gd, learning_rates_plain),
        ("SGD", test_mses_sgd, learning_rates_plain_sgd),
        ("GD (With momentum)", test_mses_gd_mom, learning_rates_plain),
        ("SGD (With momentum)", test_mses_sgd_mom, learning_rates_plain_sgd),
    )
):
    r = i // 2
    c = i % 2

    sns.heatmap(
        mse,
        ax=ax[r, c],
        xticklabels=np.round(np.log10(regularization_parameters), 2),
        yticklabels=np.round(np.log10(learning_rates), 2),
        vmax=10,
    )
    ax[r, c].set_title(title)
    ax[r, c].set_xlabel("log10($\\lambda$)")
    ax[r, c].set_ylabel("log10($\\eta$)")

fig.subplots_adjust(hspace=0.6, wspace=0.3, bottom=0.1)
fig.savefig(f"{fig_directory}/optimizers_ridge_gd_sgd.png", dpi=196)

# And a corresponding one for adagrad.
fig, ax = plt.subplots(2, 2)
for i, (title, mse) in enumerate(
    (
        ("Adagrad GD", test_mses_adagrad_gd),
        ("Adagrad SGD", test_mses_adagrad_sgd),
        ("Adagrad GD (With momentum)", test_mses_adagrad_gd_mom),
        ("Adagrad SGD (With momentum)", test_mses_adagrad_sgd_mom),
    )
):
    r = i // 2
    c = i % 2

    sns.heatmap(
        mse,
        ax=ax[r, c],
        xticklabels=np.round(np.log10(regularization_parameters), 2),
        yticklabels=np.round(np.log10(learning_rates_adagrad), 2),
        vmax=10,
    )
    ax[r, c].set_title(title)

fig.subplots_adjust(hspace=0.6, wspace=0.3, bottom=0.1)
fig.savefig(f"{fig_directory}/optimizers_ridge_adagrad.png", dpi=196)
