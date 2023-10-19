from optimizer import OptimizerGD, OptimizerAdagrad, OptimizerAdam, OptimizerRMSProp
from data import *

# Calculate some sensible learning_rate
hesse = 2 * X.T @ X
eigvalues, _ = np.linalg.eig(hesse)
learning_rate = 1.0 / np.max(eigvalues)


# We illustrate some decay function and use this when using stochastic gradient descent.
class DecayFunctionExample:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def __call__(self, t):
        return self.t0 / (t + self.t1)


somedecayfunc = DecayFunctionExample(1.0, 10)


# Optimizers
o = OptimizerGD(X, y, cost=cost_mse_ols)
o_sgd = OptimizerGD(X, y, cost=cost_mse_ols, timedecay=somedecayfunc)
o_mom = OptimizerGD(X, y, cost=cost_mse_ols, use_momentum=True)
o_mom_sgd = OptimizerGD(
    X,
    y,
    cost=cost_mse_ols,
    use_momentum=True,
    timedecay=somedecayfunc,
)

epochs = 100
sgd_size = 32
theta_gd = o.optimize(epochs, learning_rate)
theta_sgd = o_sgd.optimize_stochastic(epochs, learning_rate * 200, sgd_size)
theta_gd_mom = o_mom.optimize(epochs, learning_rate)
theta_sgd_mom = o_mom_sgd.optimize_stochastic(epochs, learning_rate * 200, sgd_size)

print(f"Analytical: {analytical}")
print(f"Without momentum\ngd: {theta_gd}\nsgd: {theta_sgd}")
print(f"With momentum\ngd: {theta_gd_mom}\nsgd: {theta_sgd_mom}")


def cost_grad_mse_ols(X, y, theta):
    return 2 * X.T @ (X @ theta - y)


o = OptimizerGD(X, y, cost_grad=cost_grad_mse_ols)
o_mom = OptimizerGD(X, y, cost_grad=cost_grad_mse_ols, use_momentum=True)
theta_gd = o.optimize(epochs, learning_rate)
theta_gd_mom = o_mom.optimize(epochs, learning_rate)
print(
    f"Without momentum gd (analytical gradient): {theta_gd}\nWith momentum gd (analytical gradient): {theta_gd_mom}"
)

# Optimizers
o_adagrad = OptimizerAdagrad(X, y, cost=cost_mse_ols)
o_adagrad_mom = OptimizerAdagrad(X, y, cost=cost_mse_ols, use_momentum=True)

# Estimates
adagrad_lr = learning_rate * 20_000
theta_adagrad = o_adagrad.optimize(epochs, adagrad_lr)
theta_adagrad_stoch = o_adagrad.optimize_stochastic(epochs, adagrad_lr, sgd_size)
theta_adagrad_mom = o_adagrad_mom.optimize(epochs, adagrad_lr)
theta_adagrad_mom_stoch = o_adagrad_mom.optimize_stochastic(
    epochs, adagrad_lr, sgd_size
)

print(f"Analytical: {analytical}")
print(f"Without momentum\nnon-stoch.: {theta_adagrad}\nstoch.: {theta_adagrad_stoch}")
print(
    f"With momentum\nnon-stoch.: {theta_adagrad_mom}\nstoch.: {theta_adagrad_mom_stoch}"
)

# Optimizers
o_rmsprop = OptimizerRMSProp(X, y, cost=cost_mse_ols, rmsprop_decay=0.99)
o_adam = OptimizerAdam(X, y, cost=cost_mse_ols)

# Estimates
theta_rmsprop = o_rmsprop.optimize(epochs, 1.0)
theta_adam = o_adam.optimize(epochs, 100.0)
theta_rmsprop_stoch = o_rmsprop.optimize_stochastic(epochs, 1.0, sgd_size)
theta_adam_stoch = o_adam.optimize_stochastic(epochs, 100.0, sgd_size)

print(f"Theta (rmsprop): {theta_rmsprop}")
print(f"Theta (rmsprop stochastic): {theta_rmsprop_stoch}")
print(f"Theta (adam): {theta_adam}")
print(f"Theta (adam stochastic): {theta_adam_stoch}")
