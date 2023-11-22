from generate_data import load_data_creditcard
import numpy as np
import tensorflow as tf

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


lmbda_grid = 10 ** np.linspace(-5, 0, 6)
lr_grid = 10 ** np.linspace(-5, 0, 6)
optimizers = [
    tf.optimizers.Adagrad,
    tf.optimizers.Adam,
    tf.optimizers.RMSprop,
    tf.optimizers.SGD,
]
# We have so much data here that in order for this to not take a crazy long
# time, we only run one epoch for model selection.
n_epochs = 1

# Collect the accuracy metrics in a 3D array.
val_accuracies = np.zeros((len(optimizers), len(lr_grid), len(lmbda_grid)))

for i, Optimizer in enumerate(optimizers):
    for j, eta in enumerate(lr_grid):
        for k, lmbda in enumerate(lmbda_grid):
            o = Optimizer(eta)
            r = tf.keras.regularizers.L2(lmbda)

            clf = generate_model(optimizer=o, regularizer=r)
            clf.fit(X_train, y_train)

            y_val_pred = np.array(clf.predict(X_val) > 0.5, dtype=int)
            val_accuracies[i, j, k] = np.mean(y_val_pred == y_val)
