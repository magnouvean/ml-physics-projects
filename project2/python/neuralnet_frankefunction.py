import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from p2libs import NeuralNet, SchedulerConstant, mse, sse

np.random.seed(1234)


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
y = franke_function(x_1, x_2).reshape((n, 1)) + 0.1 * np.random.randn(n, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

# Scale the data:
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_epochs = 200
regularization_parameters = 10 ** (np.linspace(-6, 2, 9))
learning_rates = 10 ** (np.linspace(-7, -1, 7))
mses_train = np.zeros((len(learning_rates), len(regularization_parameters)))
r2_scores_train = np.zeros_like(mses_train)
mses_test = np.zeros_like(mses_train)
r2_scores_test = np.zeros_like(mses_train)

for i, learning_rate in enumerate(learning_rates):
    scheduler = SchedulerConstant(learning_rate, use_momentum=True)
    for j, regularization_parameter in enumerate(regularization_parameters):
        nn = NeuralNet(
            (2, 100, 100, 1),
            scheduler=scheduler,
            cost=sse,
            lmbda=regularization_parameter,
        )
        nn.fit(X_train, y_train, epochs=n_epochs, sgd_size=16)
        y_train_pred = nn.predict(X_train)
        y_test_pred = nn.predict(X_test)
        nan_in_train_pred = np.any(np.isnan(y_train_pred))
        nan_in_test_pred = np.any(np.isnan(y_test_pred))
        mses_train[i, j] = np.nan if nan_in_train_pred else mse(y_train_pred, y_train)
        mses_test[i, j] = np.nan if nan_in_test_pred else mse(y_test_pred, y_test)
        r2_scores_train[i, j] = (
            np.nan if nan_in_train_pred else r2_score(y_train, y_train_pred)
        )
        r2_scores_test[i, j] = (
            np.nan if nan_in_test_pred else r2_score(y_test, y_test_pred)
        )

for label, score_matrix in (
    ("MSE train", mses_train),
    ("MSE test", mses_test),
    ("R^2 train", r2_scores_train),
    ("R^2 test", r2_scores_test),
):
    best = np.nanmin(score_matrix) if label[:3] == "MSE" else np.nanmax(score_matrix)
    best_index = (
        np.nanargmin(score_matrix) if label[:3] == "MSE" else np.nanargmax(score_matrix)
    )
    best_lr_index = best_index // score_matrix.shape[0]
    best_rgl_index = best_index % score_matrix.shape[1]
    print(
        f"{label} best: {best}, at eta={learning_rates[best_lr_index]}, lambda={regularization_parameters[best_rgl_index]}"
    )

# Set lambda and eta to the ones that gave the best performance on the test set
# over for the MSE.
eta = learning_rates[np.nanargmin(mses_test) // mses_test.shape[0]]
lmbda = regularization_parameters[np.nanargmin(mses_test) % mses_test.shape[1]]

print("\nSklearn")
nn = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=lmbda)
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
