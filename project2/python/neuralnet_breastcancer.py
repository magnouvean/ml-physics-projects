import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from p2libs import (
    NeuralNet,
    SchedulerAdagrad,
    SchedulerAdam,
    SchedulerConstant,
    SchedulerRMSProp,
    cross_entropy,
    cross_entropy_grad,
    sigmoid,
)

# It is important to set the seed for reproducibility
np.random.seed(1234)

breast_cancer_data = pd.read_csv("../data/wisconsin_data.csv", sep=",")
X = breast_cancer_data.drop(["id", "diagnosis"], axis=1).dropna(axis=1).to_numpy()
y = (breast_cancer_data["diagnosis"] == "B").to_numpy(dtype=int).reshape(X.shape[0], 1)

# Train-test-validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4)

# We first determine the amount of hidden layers by some greedy algorithm
adam_learning_rate = 0.1
scheduler = SchedulerAdam(adam_learning_rate)
n_hidden_layers_sizes = [(100,), (10, 10), (100, 100), (100, 10), (10, 10, 10)]
accuracies = np.zeros(len(n_hidden_layers_sizes))
final_losses = np.zeros(len(n_hidden_layers_sizes))
for i, hidden_layers_size in enumerate(n_hidden_layers_sizes):
    nn = NeuralNet(
        layer_sizes=(X.shape[1], *hidden_layers_size, 1),
        cost=cross_entropy,
        cost_grad=cross_entropy_grad,
        scheduler=scheduler,
        activation_functions=sigmoid,
        output_function=sigmoid,
    )
    # We give each 100 epochs and see which one gives us the best classification accuracy
    nn.fit(
        X_train,
        y_train,
        epochs=500,
        sgd=True,
        sgd_size=16,
    )
    y_train_pred_prob = nn.predict(X_train)
    y_train_pred = nn.predict(X_train, decision_boundary=0.5)

    # Calculate metrics and fill in.
    accuracy = np.mean((y_train_pred == y_train))
    accuracies[i] = accuracy
    final_losses[i] = cross_entropy(y_train_pred_prob, y_train)

print(accuracies)
print(final_losses)

# What we will see is that the accuracies essentially are the same, which likely
# is due to them making the same predictions. Looking at this doesnt give us
# much in other words. Looking at the final losses is also very much the same.
# We see that (100, 10) gave us the best so we may as well choose this going
# forward.

# We set the sizes of the hidden layers here and move forward (this should
# probably be set automatically by
# n_hidden_layers_sizes[np.argmin(final_losses)], but I do it like this for the
# sake of readability and debugability).
best_hidden_layers = (X.shape[1], 100, 10, 1)

# We won't bother to try with different activation functions here as this
# quickly becomes very computationally expensive. We now move on to tweaking the
# learning-rate, regularization-parameter and scheduler.
n_params_each = 9
learning_rates = 10 ** (np.linspace(-6, 2, n_params_each))
regularization_parameters = 10 ** (np.linspace(-9, -1, n_params_each))
schedulers = [SchedulerConstant, SchedulerAdagrad, SchedulerAdam, SchedulerRMSProp]
train_accuracies = []
val_accuracies = []
for Scheduler in schedulers:
    print(Scheduler)
    train_accuracy = np.zeros((len(learning_rates), len(regularization_parameters)))
    val_accuracy = np.zeros_like(train_accuracy)
    for i, learning_rate in enumerate(learning_rates):
        scheduler = (
            Scheduler(learning_rate)
            if Scheduler != SchedulerConstant
            else Scheduler(learning_rate, use_momentum=True)
        )
        for j, regularization_parameter in enumerate(regularization_parameters):
            nn = NeuralNet(
                best_hidden_layers,
                cost=cross_entropy,
                cost_grad=cross_entropy_grad,
                scheduler=scheduler,
                lmbda=regularization_parameter,
                output_function=sigmoid,
            )
            nn.fit(
                X_train,
                y_train,
                epochs=200,
                sgd=True,
                sgd_size=32,
            )
            y_pred_train = nn.predict(X_train, decision_boundary=0.5)
            y_pred_val = nn.predict(X_val, decision_boundary=0.5)
            train_accuracy[i, j] = np.mean(y_pred_train == y_train)
            val_accuracy[i, j] = np.mean(y_pred_val == y_val)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

for i in range(len(train_accuracies)):
    print(f"Scheduler number {i} max train accuracy: {np.max(train_accuracies[i])}")
    print(f"Scheduler number {i} max validation accuracy: {np.max(val_accuracies[i])}")


# As we can see Adam, adagrad and rmsprop all did very good. We pick adam as our chosen one however, because it did as good as the others, and usually is the
# best option.
best_pair = np.argmax(val_accuracies[2])
best_lr_index, best_rgl_index = (
    best_pair // val_accuracies[2].shape[0],
    best_pair % val_accuracies[2].shape[1],
)
print(f"Best learning_rate for best model: {learning_rates[best_lr_index]}")
print(
    f"Best regularization parameter value for best model: {regularization_parameters[best_rgl_index]}"
)

# We now can fit the final model and evalute it on the validation set.
scheduler = SchedulerAdam(learning_rate=learning_rates[best_lr_index])
nn = NeuralNet(
    best_hidden_layers,
    cost=cross_entropy,
    cost_grad=cross_entropy_grad,
    scheduler=scheduler,
    activation_functions=sigmoid,
    output_function=sigmoid,
    lmbda=regularization_parameters[best_rgl_index],
)
nn.fit(X_train, y_train, epochs=2000, print_every=200, tol=1e-2)
y_pred_test = nn.predict(X_test, decision_boundary=0.5)
final_accuracy = np.mean(y_pred_test == y_test)
print(f"Final model accuracy: {final_accuracy}")


# We do a similar fit using tensorflow here.
import tensorflow as tf

# Define regularizer and optimizer. Here I had some problems when using the
# best learning_rate from above, so I just use the default.
l2reg = tf.keras.regularizers.l2(regularization_parameters[best_rgl_index])
o = tf.keras.optimizers.Adam()

# Create a model much alike above and train it
nn = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="sigmoid", kernel_regularizer=l2reg),
        tf.keras.layers.Dense(10, activation="sigmoid", kernel_regularizer=l2reg),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2reg),
    ]
)
nn.compile(loss="binary_crossentropy", optimizer=o, metrics=["accuracy"])
nn.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val), verbose=False)

# Evaluate the final tensorflow model finally
y_pred_test = np.array(nn.predict(X_test) > 0.5, dtype=int)
final_accuracy_tf = np.mean(y_pred_test == y_test)
print(f"Final model accuracy (tensorflow): {final_accuracy_tf}")
