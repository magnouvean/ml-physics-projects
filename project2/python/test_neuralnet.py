import numpy as np
import pytest

from p2libs import NeuralNet, SchedulerAdam, SchedulerConstant, mse, sse, sse_grad


def test_correct_dimensions():
    nn = NeuralNet((1, 2, 1), mse, SchedulerConstant(0.1))
    assert nn._weights[0].shape == (1, 2)
    assert nn._weights[1].shape == (2, 1)
    assert len(nn._biases[0]) == 2
    assert len(nn._biases[1]) == 1


def test_deep_correct_dimensions():
    nn = NeuralNet((4, 89, 10, 32, 4), mse, SchedulerConstant(0.1))
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
        SchedulerConstant(0.1),
        activation_functions=lambda x: x,
    )
    X = np.zeros(4).reshape((4, 1))
    nn._weights[0] = np.ones((1, 2))
    nn._weights[1] = np.ones((2, 1))

    X[1, 0] = 4
    X[2, 0] = 5
    X[3, 0] = -1
    hidden = X @ nn._weights[0] + nn._biases[0]
    actual = hidden @ nn._weights[1] + nn._biases[1]
    y_pred = nn.forward(X)
    assert (y_pred == actual).all()


def test_jax_cost_grad():
    nn = NeuralNet((1, 2, 1), sse, SchedulerConstant(0.1))
    y = np.array([4, 2, 3, 2.3]).reshape(4, 1)
    y_pred = np.array([3.8, 1.7, 4.1, 2.35]).reshape(4, 1)
    nn_cost_grad = nn._cost_function_grad(y_pred, y)
    analytical_cost_grad = sse_grad(y_pred, y)
    print(nn_cost_grad, analytical_cost_grad)
    assert np.linalg.norm(nn_cost_grad - analytical_cost_grad) == pytest.approx(
        0, abs=1e-6
    )


def test_jax_activation_func_der():
    some_output_function = lambda x: x**3
    some_activation_function = lambda x: x**0.5
    nn = NeuralNet(
        (1, 2, 1),
        sse,
        SchedulerConstant(0.1),
        activation_functions=some_activation_function,
        output_function=some_output_function,
    )

    # Check activation function derivatives.
    x = np.array([[1.0, 2.0], [4.0, 5.0]])
    x_der_activ = 1 / (2 * x**0.5)
    for activ_func in nn._activation_functions_der:
        x_der_activ_jax = activ_func(x)
        assert np.linalg.norm(x_der_activ - x_der_activ_jax) == pytest.approx(
            0, abs=1e-6
        )

    # Check output function derivatives.
    x_der_out = 3 * x**2
    x_der_out_jax = nn._output_function_der(x)
    assert np.linalg.norm(x_der_out - x_der_out_jax) == pytest.approx(0, abs=1e-6)


def test_backward_propagation():
    # Simple data generation.
    x1 = np.random.randn(100)
    x2 = np.random.randn(100)
    x3 = np.random.randn(100)
    X = np.zeros((len(x1), 3))
    X[:, 0] = x1
    X[:, 1] = x2
    X[:, 2] = x3
    y = x1 + x2**2 + x3**3 + 2 * x1 * x2 - 6 * x2 * x3 - 3.12 * x1 * x3
    y = y.reshape(len(y), 1)

    # Create model.
    scheduler = SchedulerAdam(0.1)
    nn = NeuralNet((3, 10, 10, 10, 1), mse, scheduler)
    nn.fit(X, y, epochs=1000)

    # Predict, gather and test mse.
    y_pred = nn.forward(X)
    nn_mse = mse(y_pred, y)

    assert nn_mse < 0.01
