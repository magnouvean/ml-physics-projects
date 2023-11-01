import numpy as np

from p2libs import *

np.random.seed(1234)

f = lambda x: 4 - 6 * x**2 + x**4
x = np.linspace(-2, 2, 1001)
X = np.c_[np.ones(len(x)), x, x**2, x**3, x**4]
y = f(x)

analytical = np.linalg.pinv(X.T @ X) @ X.T @ y


def cost_mse_ols(X, y, theta):
    a = y - X @ theta
    return np.sum(a**2)


# Calculate some sensible learning_rate
hesse = 2 * X.T @ X
eigvalues, _ = np.linalg.eig(hesse)
learning_rate = 1.0 / np.max(eigvalues)
adagrad_lr = learning_rate * 20_000


# We illustrate some decay function and use this when using stochastic gradient descent.
class DecayFunctionExample:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def __call__(self, t):
        return self.t0 / (t + self.t1)


somedecayfunc = DecayFunctionExample(1.0, 10)

# Schedulers
scheduler_const_gd = SchedulerConstant(learning_rate)
scheduler_const_gd_mom = SchedulerConstant(learning_rate, use_momentum=True)
scheduler_const_sgd = SchedulerConstant(learning_rate * 200, timedecay=somedecayfunc)
scheduler_const_sgd_mom = SchedulerConstant(
    learning_rate * 200, timedecay=somedecayfunc, use_momentum=True
)
scheduler_adagrad = SchedulerAdagrad(adagrad_lr)
scheduler_adagrad_mom = SchedulerAdagrad(adagrad_lr, use_momentum=True)
scheduler_rmsprop = SchedulerRMSProp(1.0)
scheduler_adam = SchedulerAdam(100.0)

# Optimizers
o = Optimizer(X, y, scheduler_const_gd, cost=cost_mse_ols)
o_sgd = Optimizer(X, y, scheduler_const_sgd, cost=cost_mse_ols)
o_mom = Optimizer(X, y, scheduler_const_gd_mom, cost=cost_mse_ols)
o_mom_sgd = Optimizer(
    X,
    y,
    scheduler_const_sgd_mom,
    cost=cost_mse_ols,
)

epochs = 100
sgd_size = 32
theta_gd = o.optimize(epochs)
theta_sgd = o_sgd.optimize_stochastic(epochs, sgd_size)
theta_gd_mom = o_mom.optimize(epochs)
theta_sgd_mom = o_mom_sgd.optimize_stochastic(epochs, sgd_size)

print(f"Analytical: {analytical}")
print(f"Without momentum\ngd: {theta_gd}\nsgd: {theta_sgd}")
print(f"With momentum\ngd: {theta_gd_mom}\nsgd: {theta_sgd_mom}")


def cost_grad_mse_ols(X, y, theta):
    return 2 * X.T @ (X @ theta - y)


o = Optimizer(X, y, scheduler_const_gd, cost_grad=cost_grad_mse_ols)
o_mom = Optimizer(X, y, scheduler_const_gd_mom, cost_grad=cost_grad_mse_ols)
theta_gd = o.optimize(epochs)
theta_gd_mom = o_mom.optimize(epochs)
print(
    f"Without momentum gd (analytical gradient): {theta_gd}\nWith momentum gd (analytical gradient): {theta_gd_mom}"
)

# Optimizers
o_adagrad = Optimizer(X, y, scheduler_adagrad, cost=cost_mse_ols)
o_adagrad_mom = Optimizer(X, y, scheduler_adagrad_mom, cost=cost_mse_ols)

# Estimates
theta_adagrad = o_adagrad.optimize(epochs)
theta_adagrad_stoch = o_adagrad.optimize_stochastic(epochs, sgd_size)
theta_adagrad_mom = o_adagrad_mom.optimize(epochs)
theta_adagrad_mom_stoch = o_adagrad_mom.optimize_stochastic(epochs, sgd_size)

print(f"Analytical: {analytical}")
print(f"Without momentum\nnon-stoch.: {theta_adagrad}\nstoch.: {theta_adagrad_stoch}")
print(
    f"With momentum\nnon-stoch.: {theta_adagrad_mom}\nstoch.: {theta_adagrad_mom_stoch}"
)

# Optimizers
o_rmsprop = Optimizer(X, y, scheduler_rmsprop, cost=cost_mse_ols)
o_adam = Optimizer(X, y, scheduler_adam, cost=cost_mse_ols)

# Estimates
theta_rmsprop = o_rmsprop.optimize(epochs)
theta_adam = o_adam.optimize(epochs)
theta_rmsprop_stoch = o_rmsprop.optimize_stochastic(epochs, sgd_size)
theta_adam_stoch = o_adam.optimize_stochastic(epochs, sgd_size)

print(f"Theta (rmsprop): {theta_rmsprop}")
print(f"Theta (rmsprop stochastic): {theta_rmsprop_stoch}")
print(f"Theta (adam): {theta_adam}")
print(f"Theta (adam stochastic): {theta_adam_stoch}")
