import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from p2libs import (NeuralNet, SchedulerAdagrad, SchedulerAdam,
                    SchedulerConstant, SchedulerRMSProp, cross_entropy,
                    cross_entropy_grad, fig_directory, lrelu, relu, sigmoid)

# It is important to set the seed for reproducibility
np.random.seed(1234)

X, y = load_breast_cancer(return_X_y=True)
y = y.reshape(len(y), 1)

# Train-test-validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4)

# Scale the data using standard-scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# We first determine the amount of hidden layers which gives fastest
# convergence on the training data.
learning_rates = 10 ** (np.linspace(-7, 2, 10))
n_hidden_layers_sizes = [
    (100,),
    (10, 10),
    (100, 100),
    (50, 10),
    (10, 10, 10),
]

accuracies = np.zeros((len(n_hidden_layers_sizes), len(learning_rates)))
final_losses = np.zeros_like(accuracies)
for i, hidden_layers_size in enumerate(n_hidden_layers_sizes):
    for j, learning_rate in enumerate(learning_rates):
        scheduler = SchedulerAdam(learning_rate)
        nn = NeuralNet(
            layer_sizes=(X.shape[1], *hidden_layers_size, 1),
            cost=cross_entropy,
            cost_grad=cross_entropy_grad,
            scheduler=scheduler,
            activation_functions=sigmoid,
            output_function=sigmoid,
        )
        # We give each 100 epochs and see which one gives us the best
        # classification accuracy.
        nn.fit(
            X_train,
            y_train,
            epochs=200,
            sgd=True,
            sgd_size=16,
        )
        y_train_pred_prob = nn.predict(X_train)
        y_train_pred = nn.predict(X_train, decision_boundary=0.5)

        # Calculate metrics and fill in.
        accuracy = np.mean((y_train_pred == y_train))
        accuracies[i, j] = accuracy
        final_losses[i, j] = cross_entropy(y_train_pred_prob, y_train)

# Show best accuracy/end of training loss for each layer size and which
# learning_rate this was acquired by.
print("========Hidden layers========")
print("Best accuracies:")
for n_hidden_layers, acc in zip(n_hidden_layers_sizes, accuracies):
    print(
        f"{n_hidden_layers}: {np.nanmax(acc)}, eta={learning_rates[np.nanargmax(acc)]}"
    )

print("\nEnd of training loss:")
for n_hidden_layers, loss in zip(n_hidden_layers_sizes, final_losses):
    print(
        f"{n_hidden_layers}: {np.nanmin(loss)}, eta={learning_rates[np.nanargmin(loss)]}"
    )

# We see that all of the models eventually are able to classify all the
# training-data correctly (though this is likely very overfitted). Looking at
# how quick the models have learnt however we see that the (100, 100) model has
# yielded the best result.
best_hidden_layers = (X.shape[1], 100, 100, 1)

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
            epochs=200,
            sgd=True,
            sgd_size=16,
        )
        y_train_pred_prob = nn.predict(X_train)
        y_train_pred = nn.predict(X_train, decision_boundary=0.5)

        # Calculate metrics and fill in.
        accuracy = np.mean((y_train_pred == y_train))
        accuracies[i, j] = accuracy
        final_losses[i, j] = cross_entropy(y_train_pred_prob, y_train)

print("\n========Activation Functions========")
print("Accuracies:")
for activ_func, accs in zip(activation_functions, accuracies):
    print(
        f"{activ_func} max acc: {np.nanmax(accs)}, lr={learning_rates[np.nanargmax(accuracies)]}"
    )
print("\nEnd of training loss:")
for activ_func, losses in zip(activation_functions, final_losses):
    print(
        f"{activ_func}: {np.nanmin(losses)}, lr={learning_rates[np.nanargmin(losses)]}"
    )


# What we see is that here actually sigmoid did give fastest convergence. We
# therefore move on with this as our activation function for the hidden layers.

# After having chosen the hidden layers and activation function we now test the
# different schedulers on the various learning-rates and
# regularization_parameters.
schedulers = [SchedulerConstant, SchedulerAdagrad, SchedulerAdam, SchedulerRMSProp]

# We then have to define a grid of regularization parameters to test the
# accuracy on.
regularization_parameters = 10 ** (np.linspace(-9, 0, len(learning_rates)))

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
    print(f"{scheduler_names[i]} max train accuracy: {np.nanmax(train_accuracies[i])}")
    print(
        f"{scheduler_names[i]} max validation accuracy: {np.nanmax(val_accuracies[i])}"
    )

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
        ax.set_xlabel("log10($\eta$)")
        ax.set_ylabel("log10($\lambda$)")

# Adjust the margins between subplots so that the axis labels become readable.
fig_train.subplots_adjust(hspace=0.8, wspace=0.5, bottom=0.15)
fig_val.subplots_adjust(hspace=0.8, wspace=0.5, bottom=0.15)
# And save the figures in these directories.
fig_train.savefig(f"{fig_directory}/heatmap_train.png", dpi=196)
fig_val.savefig(f"{fig_directory}/heatmap_val.png", dpi=196)


# As we can see all of the schedulers did a very good job here, but seeing as
# Adagrad gave just about the highest accuracy, we move ahead with this for our
# final model.
best_pair = np.argmax(val_accuracies[1])
best_lr_index, best_rgl_index = (
    best_pair // val_accuracies[1].shape[0],
    best_pair % val_accuracies[1].shape[1],
)
best_lr = learning_rates[best_lr_index]
best_rgl = regularization_parameters[best_rgl_index]
print(f"Best learning_rate for adagrad: {best_lr}")
print(f"Best regularization parameter value for adagrad: {best_rgl}")

# We now can fit the final model and evaluate it on the validation set. We may
# now are also done with using the validation data and we may then merge this
# together into one (we don't need the X_val anymore so we may as well use it
# for training the model, as more data is almost always better).
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])
scheduler = SchedulerAdagrad(learning_rate=best_lr)
nn = NeuralNet(
    best_hidden_layers,
    cost=cross_entropy,
    cost_grad=cross_entropy_grad,
    scheduler=scheduler,
    activation_functions=sigmoid,
    output_function=sigmoid,
    lmbda=regularization_parameters[best_rgl_index],
)
# We now give the neural net more epochs just in case it needs it for
# convergence (though it probably doesn't).
nn.fit(X_train, y_train, epochs=2000, print_every=200)
y_pred_test = nn.predict(X_test, decision_boundary=0.5)
final_accuracy = np.mean(y_pred_test == y_test)
print(f"Final model accuracy: {final_accuracy}")

# And finally create a confusion matrix for this final model.
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
plt.savefig(f"{fig_directory}/confusion_matrix_nn.png")


# We do a similar fit using tensorflow here.
import tensorflow as tf

# Define regularizer and optimizer. Here I had some problems when using the
# best learning_rate from above, so I just use the default.
l2reg = tf.keras.regularizers.l2(best_rgl)
o = tf.keras.optimizers.experimental.Adagrad(learning_rate=best_lr)

# Create a model much alike above and train it.
nn = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation="sigmoid", kernel_regularizer=l2reg),
        tf.keras.layers.Dense(10, activation="sigmoid", kernel_regularizer=l2reg),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2reg),
    ]
)
nn.compile(loss="binary_crossentropy", optimizer=o, metrics=["accuracy"])
nn.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val), verbose=False)

# Evaluate the final tensorflow model finally.
y_pred_test = np.array(nn.predict(X_test) > 0.5, dtype=int)
final_accuracy_tf = np.mean(y_pred_test == y_test)
print(f"Final model accuracy (tensorflow): {final_accuracy_tf}")

# We may also create confusion for this model as well
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
disp.ax_.set_title("Confusion matrix tensorflow")
plt.savefig(f"{fig_directory}/confusion_matrix_tensorflow.png")
