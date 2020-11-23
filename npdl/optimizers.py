import numpy as np

class Optimizer(object):
    def initiate_parameters(self, layers: list):
        pass

    def update_parameters(self, layers: list):
        pass

    def update_optimizer(self, metrics: dict):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layers: list):
        for layer in layers:
            layer.W = layer.W - self.learning_rate * layer.dW
            layer.b = layer.b - self.learning_rate * layer.db
        return True


class Momentum(Optimizer):
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta

    def initiate_parameters(self, layers: list):
        for layer in layers:
            layer.V_dW = np.zeros(layer.W.shape)
            layer.V_db = np.zeros(layer.b.shape)
        return True

    def update_parameters(self, layers: list):
        for layer in layers:
            layer.V_dW = self.beta * layer.V_dW + (1 - self.beta) * layer.dW
            layer.V_db = self.beta * layer.V_db + (1 - self.beta) * layer.db

            layer.W = layer.W - self.learning_rate * layer.V_dW
            layer.b = layer.b - self.learning_rate * layer.V_db
        return True


class RMSprop(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-8

    def initiate_parameters(self, layers: list):
        for layer in layers:
            layer.S_dW = np.zeros(layer.W.shape)
            layer.S_db = np.zeros(layer.b.shape)
        return True

    def update_parameters(self, layers: list):
        for layer in layers:
            layer.S_dW = self.beta * layer.S_dW + (1 - self.beta) * np.square(
                layer.dW)
            layer.S_db = self.beta * layer.S_db + (1 - self.beta) * np.square(
                layer.db)

            layer.W = layer.W - self.learning_rate * (
                layer.dW / (np.sqrt(layer.S_dW) + self.epsilon))
            layer.b = layer.b - self.learning_rate * (
                layer.db / (np.sqrt(layer.S_db) + self.epsilon))
        return True


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-7
        self.counter = 0

    def initiate_parameters(self, layers: list):
        self.counter = 0
        for layer in layers:
            layer.V_dW = np.zeros(layer.W.shape)
            layer.V_db = np.zeros(layer.b.shape)
            layer.S_dW = np.zeros(layer.W.shape)
            layer.S_db = np.zeros(layer.b.shape)

            layer.V_corrected_dW = np.zeros(layer.W.shape)
            layer.V_corrected_db = np.zeros(layer.b.shape)
            layer.S_corrected_dW = np.zeros(layer.W.shape)
            layer.S_corrected_db = np.zeros(layer.b.shape)
        return True

    def update_parameters(self, layers: list):
        for layer in layers:
            layer.V_dW = self.beta_1 * layer.V_dW + (1 -
                                                     self.beta_1) * layer.dW
            layer.V_db = self.beta_1 * layer.V_db + (1 -
                                                     self.beta_1) * layer.db
            layer.S_dW = self.beta_2 * layer.S_dW + (
                1 - self.beta_2) * np.square(layer.dW)
            layer.S_db = self.beta_2 * layer.S_db + (
                1 - self.beta_2) * np.square(layer.db)

            # Apply bias correction
            momentum_correction = 1 / (1 - self.beta_1**(self.counter + 1))
            rmsprop_correction = 1 / (1 - self.beta_2**(self.counter + 1))

            layer.V_corrected_dW = layer.V_dW * momentum_correction
            layer.V_corrected_db = layer.V_db * momentum_correction
            layer.S_corrected_dW = layer.S_dW * rmsprop_correction
            layer.S_corrected_db = layer.S_db * rmsprop_correction

            layer.W = layer.W - self.learning_rate * (
                layer.V_corrected_dW /
                (np.sqrt(layer.S_corrected_dW) + self.epsilon))
            layer.b = layer.b - self.learning_rate * (
                layer.V_corrected_db /
                (np.sqrt(layer.S_corrected_db) + self.epsilon))
        self.counter += 1
        return True