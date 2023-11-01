import pandas as pd
from sklearn.model_selection import train_test_split
from p2libs import NeuralNet, SchedulerAdam, sigmoid, cross_entropy, cross_entropy_grad

breast_cancer_data = pd.read_csv("../data/wisconsin_data.csv", sep=",")
X = breast_cancer_data.drop(["id", "diagnosis"], axis=1).dropna(axis=1).to_numpy()
y = (breast_cancer_data["diagnosis"] == "B").to_numpy().reshape(X.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

scheduler = SchedulerAdam(0.001)
nn = NeuralNet(
    layer_sizes=(X.shape[1], 100, 100, 1),
    cost=cross_entropy,
    cost_grad=cross_entropy_grad,
    scheduler=scheduler,
    activation_functions=sigmoid,
    output_function=sigmoid,
)
nn.fit(X_train, y_train, print_every=100, sgd=True, sgd_size=32, epochs=1000)
y_train_pred = nn.forward(X_train)
