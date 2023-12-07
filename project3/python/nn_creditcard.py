import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from generate_data import load_data_creditcard

X_train, y_train, X_val, y_val, X_test, y_test = load_data_creditcard()


def generate_model(
    activation="relu",
    hidden_layer_sizes=(100, 100),
    optimizer="adam",
    regularizer=tf.keras.regularizers.L2(0.01),
):
    hidden_layers = [
        tf.keras.layers.Dense(
            hidden_layer_size, activation=activation, kernel_regularizer=regularizer
        )
        for hidden_layer_size in hidden_layer_sizes
    ]
    clf = tf.keras.Sequential(
        [
            *hidden_layers,
            tf.keras.layers.Dense(
                1, activation="sigmoid", kernel_regularizer=regularizer
            ),
        ]
    )
    clf.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return clf


lmbda_grid = 10 ** np.linspace(-8, -2, 7)
lr_grid = 10 ** np.linspace(-4, 1, 11)
optimizers = [
    tf.optimizers.Adagrad,
    tf.optimizers.Adam,
    tf.optimizers.RMSprop,
    tf.optimizers.SGD,
]

# Collect the accuracy metrics in a 3D array.
val_accuracies = np.zeros((len(optimizers), len(lr_grid), len(lmbda_grid)))

# Because training takes a long time for each epoch on the full data we limit
# our data somewhat when doing model selection. When we have found our final
# model we will of train on the full data. Thus our best model will be based
# just on a smaller training-set, but the hyperparamters which gives the best
# results in this case should be similar to the optimial hyperparameters for
# the full model (hopefully anyway).
X_train_smaller, y_train_smaller = X_train[:20_000], y_train[:20_000]
X_val_smaller, y_val_smaller = X_val[:20_000], y_val[:20_000]

for i, Optimizer in enumerate(optimizers):
    for j, eta in enumerate(lr_grid):
        for k, lmbda in enumerate(lmbda_grid):
            o = Optimizer(eta)
            r = tf.keras.regularizers.L2(lmbda)

            clf = generate_model(optimizer=o, regularizer=r)
            # We have so much data here that in order for this to not take a crazy long
            # time, we only run one epoch for model selection.
            clf.fit(X_train_smaller, y_train_smaller, verbose=0)

            y_val_pred = np.array(
                clf.predict(X_val_smaller, verbose=0) > 0.5, dtype=int
            )
            val_accuracies[i, j, k] = np.mean(y_val_pred == y_val_smaller)


fig, ax = plt.subplots(2, 2)
for i, val_accuracy_mat in enumerate(val_accuracies):
    r = i // 2
    c = i % 2
    sns.heatmap(val_accuracy_mat, ax=ax[r, c])

ax[0, 0].set_title("AdaGrad")
ax[0, 1].set_title("Adam")
ax[1, 0].set_title("RMSProp")
ax[1, 1].set_title("SGD")
fig.savefig("../figures/nn_test_accuracy_hm.png", dpi=196)

print("====Model performances====")
print(f"AdaGrad best: {np.nanmax(val_accuracies[0])}")
print(f"Adam best: {np.nanmax(val_accuracies[1])}")
print(f"RMSProp best: {np.nanmax(val_accuracies[2])}")
print(f"SGD best: {np.nanmax(val_accuracies[3])}")

# We now will keep using the RMSProp as our optimizer, but check out some
# different activation functions.
# See https://www.tensorflow.org/api_docs/python/tf/keras/activations for a
# list of allowed activation functions.
activation_functions = ("relu", "sigmoid", "swish", "tanh", "elu")
val_accuracies = np.zeros((len(activation_functions), len(lr_grid), len(lmbda_grid)))

for i, activation_function in enumerate(activation_functions):
    for j, eta in enumerate(lr_grid):
        for k, lmbda in enumerate(lmbda_grid):
            o = tf.keras.optimizers.RMSprop(eta)
            clf = generate_model(
                activation=activation_function,
                optimizer=o,
                regularizer=tf.keras.regularizers.L2(lmbda),
            )
            clf.fit(X_train_smaller, y_train_smaller, verbose=False)
            y_val_pred = np.array(
                clf.predict(X_val_smaller, verbose=0) > 0.5, dtype=int
            )
            val_accuracies[i, j, k] = np.mean(y_val_pred == y_val_smaller)

print("====Activation functions====")
for activation_function, val_accuracy_mat in zip(activation_functions, val_accuracies):
    print(f"{activation_function}: {np.nanmax(val_accuracy_mat)}")

# We see that relu does pretty much the best and in addition is very quick, so
# this is the one we will continue with. We also determine the best
# learning-rate and regularization parameter for RMSProp using relu.
best_eta_index = np.nanargmax(val_accuracies[0]) // len(lmbda_grid)
best_lmbda_index = np.nanargmax(val_accuracies[0]) % len(lmbda_grid)
best_eta = lr_grid[best_eta_index]
best_lmbda = lmbda_grid[best_lmbda_index]
print(
    f"\n\nRMSProp relu best hyperparams (100, 100): eta=10^{np.log10(best_eta)}, lambda=10^{np.log10(best_lmbda)}"
)

# We then try a couple different activation functions and network sizes. We
# have:
hidden_layers_sizes = (
    (10,),
    (10, 10, 10),
    (100, 100),
    (50, 50),
    (100, 100, 100, 100),
    (10, 10, 10, 10),
    (100, 100, 100, 100, 100),
    (1000, 100, 10),
    (1000, 1000),
    (1000, 1000, 1000),
    (40, 100, 30),
)
val_accuracies = np.zeros((len(hidden_layers_sizes), len(lmbda_grid)))

for i, hidden_layers_size in enumerate(hidden_layers_sizes):
    for j, lmbda in enumerate(lmbda_grid):
        o = tf.keras.optimizers.RMSprop(best_eta)
        clf = generate_model(
            activation="relu",
            hidden_layer_sizes=hidden_layers_size,
            optimizer=o,
            regularizer=tf.keras.regularizers.L2(lmbda),
        )
        clf.fit(X_train_smaller, y_train_smaller, verbose=False)
        y_val_pred = np.array(clf.predict(X_val_smaller, verbose=0) > 0.5, dtype=int)
        val_accuracies[i, j] = np.mean(y_val_pred == y_val_smaller)

print(
    {
        hidden_layers_size: np.max(val_accuracy)
        for hidden_layers_size, val_accuracy in zip(hidden_layers_sizes, val_accuracies)
    }
)


# What we see is that the bigger networks seem to perform better here, but of
# course these are more heavy to train. Keep also in mind that we here are
# training on a much smaller dataset than we will on the final model, so we
# will keep using (100, 100) as the layer sizes since this one is quite easy to
# train and seems to perform well anyhow. We now try some different activation
# functions.

# We then are ready to train our final model this time on the full data and
# multiple epochs.
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])
clf = generate_model(
    optimizer=tf.keras.optimizers.RMSprop(best_eta),
    regularizer=tf.keras.regularizers.L2(best_lmbda),
)

# Ideally we don't want to train 100 epochs as that will be a waste of time
# (and may even give us a more overfitted model), so we will use early stopping
# callback here.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=3, min_delta=1e-3
)
clf.fit(X_train, y_train, epochs=100, callbacks=[early_stopping], verbose=False)
clf.save("../models/nn_creditcard")

y_test_pred_proba = clf.predict(X_test, verbose=0)
y_test_pred = np.array(y_test_pred_proba > 0.5, dtype=int)
final_accuracy = np.mean(y_test_pred == y_test)

print("====Final model accuracy====")
print(final_accuracy)

# Confusion matrix and roc curve models
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, colorbar=False)
plt.savefig("../figures/nn_final_confusion_mat.png", dpi=196)

RocCurveDisplay.from_predictions(y_test, y_test_pred_proba)
plt.savefig("../figures/nn_final_roc_curve.png", dpi=196)
