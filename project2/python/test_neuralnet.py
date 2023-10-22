import pytest
from neuralnet import *


def test_correct_dimensions():
    nn = NeuralNet((1, 2, 1), mse, mse_der)
    assert nn._weights[0].shape == (1, 2)
    assert nn._weights[1].shape == (2, 1)
    assert len(nn._biases[0]) == 2
    assert len(nn._biases[1]) == 1


def test_deep_correct_dimensions():
    nn = NeuralNet((4, 89, 10, 32, 4), mse, mse_der)
    assert nn._weights[0].shape == (4, 89)
    assert nn._weights[1].shape == (89, 10)
    assert nn._weights[2].shape == (10, 32)
    assert nn._weights[3].shape == (32, 4)
    assert len(nn._biases[0]) == 89
    assert len(nn._biases[1]) == 10
    assert len(nn._biases[2]) == 32
    assert len(nn._biases[3]) == 4


def test_forward_propagration():
    nn = NeuralNet(
        (1, 2, 1),
        mse,
        mse_der,
        activation_functions=lambda x: x,
        activation_functions_der=lambda x: 1,
    )
    X = np.zeros(4).reshape((4, 1))
    nn._weights[0] = np.ones((1, 2))
    nn._weights[1] = np.ones((2, 1))

    X[1, 0] = 4
    X[2, 0] = 5
    X[3, 0] = -1
    hidden = X @ nn._weights[0] + nn._biases[0]
    hidden = hidden @ nn._weights[1] + nn._biases[1]
    y_pred = nn.forward(X)
    assert y_pred == actual
