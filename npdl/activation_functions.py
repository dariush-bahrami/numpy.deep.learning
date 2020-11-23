def sigmoid(z):
    # negative_overflow = np.where(z<=-709, 0, 0)
    okay_range = np.where((-709 < z) & (z < 709), 1 / (1 + np.exp(-z)), 0)
    positive_overflow = np.where(z >= 709, 1, 0)
    return okay_range + positive_overflow


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z, leak_grad=0):
    return np.maximum(leak_grad * z, z)
"""This module contain most used activation functions"""
import numpy as np

def relu_derivative(z, leak_grad=0):
    return np.where(z <= 0, leak_grad, 1)


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(z):
    return 1 - tanh(z)**2


def softmax(Z, epsilon=1e-8):
    Z = Z - np.max(Z)
    exp_Z = np.exp(Z)
    return exp_Z / (np.sum(exp_Z, axis=0) + epsilon)


def softmax_derivative(Z):
    return 1


def identity(Z):
    return Z


def identity_derivative(Z):
    return 1