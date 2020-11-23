from activation_functions import *
import numpy as np


def get_cross_entropy_cost_function(binary=False):
    def binary_cross_entropy(Y, Y_hat, epsilon=1e-4):
        assert Y.shape == Y_hat.shape, f'{Y.shape} != {Y_hat.shape}'
        m = Y.shape[1]  # number of examples
        cost = (-1 / m) * (np.dot(Y,
                                  np.log(Y_hat + epsilon).T) + np.dot(
                                      (1 - Y),
                                      np.log(1 - Y_hat + epsilon).T))
        return cost.item()

    def binary_cross_entropy_derivative(Y, Y_hat, epsilon=1e-4):
        return ((1 - Y) / (1 - Y_hat + epsilon)) - (Y / (Y_hat + epsilon))

    def softmax_crossentropy(Y, Y_hat, epsilon=1e-4):
        assert Y.shape == Y_hat.shape, f'{Y.shape} != {Y_hat.shape}'
        Y_hat = softmax(Y_hat)
        m = Y.shape[1]
        cost = 1 / m * np.sum(-np.sum(Y * np.log(Y_hat + epsilon), axis=0))
        return cost

    def softmax_crossentropy_derivative(Y, Y_hat, epsilon=1e-7):
        Y_hat = softmax(Y_hat)
        return Y_hat - Y

    if binary:
        return binary_cross_entropy, binary_cross_entropy_derivative
    else:
        return softmax_crossentropy, softmax_crossentropy_derivative


def get_regularization_cost(layers: list, m: int, lambda_: float) -> float:
    cost = 0
    for layer in layers:
        cost += np.sum(np.square(layer.W))
    cost *= (1 / m) * (lambda_ / 2)
    return cost