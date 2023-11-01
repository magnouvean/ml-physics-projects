import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from p2libs import *

np.random.seed(314159)


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


n = 200
X = np.zeros((n, 2))
Y = np.zeros_like(X)
x_1 = np.random.uniform(0, 1, n)
x_2 = np.random.uniform(0, 1, n)
X[:, 0] = (x_1 - np.mean(x_1)) / np.std(x_1)
X[:, 1] = (x_2 - np.mean(x_2)) / np.std(x_2)
y = franke_function(x_1, x_2).reshape(n, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)


# Define the schedulers we need:
scheduler_constant = SchedulerConstant(learning_rate=0.0005, use_momentum=True)
scheduler_rmsprop = SchedulerRMSProp(learning_rate=0.0005)
scheduler_adagrad = SchedulerAdagrad(learning_rate=0.05, use_momentum=True)
scheduler_adam = SchedulerAdam(learning_rate=0.01)


n_epochs = 200
lmbda = 0.001

for method_name, scheduler in [
    ("Constant", scheduler_constant),
    ("RMSProp", scheduler_rmsprop),
    ("Adagrad", scheduler_adagrad),
    ("Adam", scheduler_adam),
]:
    nn = NeuralNet(
        (2, 100, 100, 1),
        scheduler=scheduler,
        cost=sse,
        lmbda=lmbda,
    )
    nn.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        print_every=int(n_epochs / 10),
        sgd=True,
        sgd_size=32,
    )
    y_pred_train = nn.forward(X_train)
    y_pred_test = nn.forward(X_test)
    nn.fit(X_train, y_train)
    y_pred_train = nn.forward(X_train).reshape(X_train.shape[0], 1)
    y_pred_test = nn.forward(X_test).reshape(X_test.shape[0], 1)

    print(f"\n{method_name}")
    print(f"mse, train ({method_name}): {mse(y_pred_train, y_train)}")
    print(f"mse, test ({method_name}): {mse(y_pred_test, y_test)}")

print("\nSklearn")
nn = MLPRegressor(hidden_layer_sizes=(100, 100))
nn.fit(X_train, y_train)
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)
print(f"mse, train (sklearn): {mse(y_pred_train, y_train)}")
print(f"mse, test (sklearn): {mse(y_pred_test, y_test)}")

print("\nTensorflow")
tf.random.set_seed(1234)
nn = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            100,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(lmbda),
        ),
        tf.keras.layers.Dense(
            100,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(lmbda),
        ),
        tf.keras.layers.Dense(1),
    ]
)
sgd = tf.keras.optimizers.SGD(learning_rate=0.005)
nn.compile(loss="mse", optimizer=sgd, metrics=["mse"])
nn.fit(X_train, y_train)
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)
print(f"mse, train (tensorflow): {mse(y_pred_train, y_train)}")
print(f"mse, test (tensorflow): {mse(y_pred_test, y_test)}")
