import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from p2libs import (NeuralNet, SchedulerAdagrad, SchedulerAdam,
                    SchedulerConstant, SchedulerRMSProp, cross_entropy,
                    cross_entropy_grad, lrelu, relu, sigmoid)

# It is important to set the seed for reproducibility
np.random.seed(1234)

X, y = load_breast_cancer(return_X_y=True)
y = y.reshape(len(y), 1)

# Train-test-validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4)

# We first determine the amount of hidden layers by some greedy algorithm
adam_learning_rate = 0.1
scheduler = SchedulerAdam(adam_learning_rate)
n_hidden_layers_sizes = [
    (100,),
    (10, 10),
    (100, 100),
    (50, 10),
    (10, 10, 10),
]
accuracies = np.zeros(len(n_hidden_layers_sizes))
final_losses = np.zeros_like(accuracies)
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

print(f"Accuracies:")
for n_hidden_layers, acc in zip(n_hidden_layers_sizes, accuracies):
    print(f"{n_hidden_layers}: {acc}")
print(f"\n\nEnd of training loss:")
for n_hidden_layers, loss in zip(n_hidden_layers_sizes, final_losses):
    print(f"{n_hidden_layers}: {loss}")

# What we can see is that we actually get the best results using only a single
# layer with 100 neurons (out of the ones we tested), with this being the best
# in terms of training loss at the end, and accuracy on the training data.
#
# We now set the sizes of the hidden layers here and move forward (this should
# probably be set automatically by
# n_hidden_layers_sizes[np.argmin(final_losses)], but I do it like this for the
# sake of readability and debugability).
best_hidden_layers = (X.shape[1], 100, 1)


# We define a grid of learning_rates and regularization parameters to perform a
# grid-search on.
n_params_each = 9
learning_rates = 10 ** (np.linspace(-6, 2, n_params_each))
regularization_parameters = 10 ** (np.linspace(-9, -1, n_params_each))

# We now test some activation functions for the various learning_rates and only
# using adam.
activation_functions = [sigmoid, relu, lrelu]

# Now since we for each activation function have some amount of learning_rates
# to fit the accuracies/final losses become a 2d matrix, where at element (i,
# j) is the i-th activation function and the j-th learning-rate.
accuracies = np.zeros((len(activation_functions), len(learning_rates)))
final_losses = np.zeros_like(accuracies)
for i, activ_func in enumerate(activation_functions):
    for j, learning_rate in enumerate(learning_rates):
        scheduler = SchedulerAdam(learning_rate)
        nn = NeuralNet(
            layer_sizes=best_hidden_layers,
            cost=cross_entropy,
            cost_grad=cross_entropy_grad,
            scheduler=scheduler,
            activation_functions=activ_func,
            output_function=sigmoid,
        )
        nn.fit(
            X_train,
            y_train,
            epochs=1000,
            sgd=True,
            sgd_size=16,
        )
        y_train_pred_prob = nn.predict(X_train)
        y_train_pred = nn.predict(X_train, decision_boundary=0.5)

        # Calculate metrics and fill in.
        accuracy = np.mean((y_train_pred == y_train))
        accuracies[i, j] = accuracy
        final_losses[i, j] = cross_entropy(y_train_pred_prob, y_train)

print(f"Accuracies:")
for activ_func, accs in zip(n_hidden_layers_sizes, accuracies):
    print(f"{activ_func} max acc: {accs.max()}")
print(f"\n\nEnd of training loss:")
for activ_func, losses in zip(n_hidden_layers_sizes, final_losses):
    print(f"{activ_func}: {losses.max()}")


# After having chosen the hidden layers and activation function we now test the
# different schedulers on the various learning-rates and
# regularization_parameters.
schedulers = [SchedulerConstant, SchedulerAdagrad, SchedulerAdam, SchedulerRMSProp]

# These should be lists containing matrices with element (i, j) being the
# accuracy achieved on learning_rate nr i and regularization_parameter j.
train_accuracies = []
val_accuracies = []

for Scheduler in schedulers:
    train_accuracy = np.zeros(
        (len(learning_rates), len(regularization_parameters)), dtype=float
    )
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

scheduler_names = ("SGD", "Adagrad", "Adam", "RMSProp")
for i in range(len(train_accuracies)):
    print(f"{scheduler_names[i]} max train accuracy: {train_accuracies[i].max()}")
    print(f"{scheduler_names[i]} max validation accuracy: {val_accuracies[i].max()}")

# Create some heatmaps for the parameters.
fig_train, ax_train = plt.subplots(2, 2)
fig_val, ax_val = plt.subplots(2, 2)
for i in range(4):
    r = i // 2
    c = i % 2
    sns.heatmap(
        train_accuracies[i],
        ax=ax_train[r, c],
        xticklabels=np.log10(regularization_parameters),
        yticklabels=np.log10(learning_rates),
    )
    sns.heatmap(
        val_accuracies[i],
        ax=ax_val[r, c],
        xticklabels=np.log10(regularization_parameters),
        yticklabels=np.log10(learning_rates),
    )
    for ax in (ax_train[r, c], ax_val[r, c]):
        ax.set_title(scheduler_names[i])
        ax.set_xlabel("$log10(\eta)$")
        ax.set_ylabel("$log10(\lambda)$")

# Assign the directory in which the file or shell is in. We then can use this
# to load/save files relatively.
current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# We then determine the figures directory based on this.
fig_directory = f"{os.path.dirname(current_dir)}/figures"
# Adjust the margins between subplots so that the axis labels become readable.
fig_train.subplots_adjust(hspace=0.8, wspace=0.5, bottom=0.15)
fig_val.subplots_adjust(hspace=0.8, wspace=0.5, bottom=0.15)
# And save the figures in these directories.
fig_train.savefig(f"{fig_directory}/heatmap_train.png", dpi=196)
fig_val.savefig(f"{fig_directory}/heatmap_val.png", dpi=196)


# As we can see Adam, adagrad and rmsprop all did quite good with more than 90%
# test accuracy. We pick adam for our chosen model however, because it performed
# marginally better.
best_pair = np.argmax(val_accuracies[2])
best_lr_index, best_rgl_index = (
    best_pair // val_accuracies[2].shape[0],
    best_pair % val_accuracies[2].shape[1],
)
best_lr = learning_rates[best_lr_index]
best_rgl = regularization_parameters[best_rgl_index]
print(f"Best learning_rate for adam: {best_lr}")
print(f"Best regularization parameter value for adam: {best_rgl}")

# We now can fit the final model and evaluate it on the validation set.
# We may now are also done with using the validation data and we may then merge
# this together into one (we don't need the X_val anymore so we may as well use
# it for training the model, as more data is almost always better).
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])
scheduler = SchedulerAdam(learning_rate=best_lr)
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
l2reg = tf.keras.regularizers.l2(best_rgl)
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
