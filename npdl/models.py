from structure import *
from optimizers import *
from tools import *
from activation_functions import *


class Classifier(object):
    def __init__(self, model_structure: ModelStructure, binary=False):
        self.input_features = model_structure.layers[0].units_number
        self.layers = initialize(model_structure.layers)
        self.layers_backup = copy.deepcopy(self.layers)
        self.cost_function = get_cross_entropy_cost_function(binary=binary)[0]
        self.cost_function_derivative = get_cross_entropy_cost_function(
            binary=binary)[1]
        self.accuracy_func = get_accuracy_function(binary=binary)
        self.predict_function = get_predict_function(apply_softmax=not binary)
        self.metrics = {
            'costs': [],
            'accuracies': [],
            'validation_costs': [],
            'validation_accuracies': [],
            'total_trained_epochs': 0,
            'current_trained_epochs': 0,
        }

    def undo_update(self):
        self.layers = copy.deepcopy(self.layers_backup)
        self.metrics['costs'].pop()
        self.metrics['accuracies'].pop()
        self.metrics['validation_costs'].pop()
        self.metrics['validation_accuracies'].pop()
        self.metrics['total_trained_epochs'] -= 1
        self.metrics['current_trained_epochs'] -= 1
        print('Last Successful Parameters Recovered')

    def feed_forward(self, X: np.ndarray):
        assert X.shape[0] == self.input_features
        A_prev = X
        for layer in self.layers:
            Z = np.dot(layer.W, A_prev) + layer.b
            A = layer.activation_function(Z)
            A_prev = A
        Y_hat = A_prev
        return Y_hat

    def feed_forward_train(self, X: np.ndarray):
        assert X.shape[0] == self.input_features
        A_prev = X
        for layer in self.layers:
            layer.A_previous = A_prev
            layer.Z = np.dot(layer.W, A_prev) + layer.b
            A_raw = layer.activation_function(
                layer.Z)  # Before applying dropout
            layer.D = np.random.rand(
                *A_raw.shape) < layer.keep_prob  # Dropout Mask
            layer.A = (A_raw * layer.D) / layer.keep_prob
            A_prev = layer.A
        Y_hat = A_prev
        return Y_hat

    def back_propagate(self, cost_derivative, lambda_):
        dA_prev = cost_derivative
        m = self.layers[0].Z.shape[1]  # number of examples
        for layer in self.layers[::-1]:
            layer.dA = (dA_prev * layer.D) / layer.keep_prob
            layer.dZ = layer.dA * layer.activation_function_derivative(layer.Z)
            layer.dW = (1 / m) * np.dot(
                layer.dZ, layer.A_previous.T) + (lambda_ / m) * layer.W
            layer.db = 1 / m * np.sum(layer.dZ, axis=1, keepdims=True)
            dA_prev = np.dot(layer.W.T, layer.dZ)
        return True

    def fit_minibatch(self, mini_X, mini_Y, lambda_, optimizer):
        mini_Y_hat = self.feed_forward_train(mini_X)
        cost_derivative = self.cost_function_derivative(mini_Y, mini_Y_hat)
        self.back_propagate(cost_derivative, lambda_)
        optimizer.update_parameters(self.layers)
        return True

    def fit(self,
            X,
            Y,
            epochs,
            batch_size,
            optimizer,
            lambda_=0,
            validation_data=None,
            metrics_printer=None):

        # Assert Shapes
        assert X.shape[1] == Y.shape[1]
        m = X.shape[1]
        if validation_data:
            assert type(validation_data) == tuple
            assert type(validation_data[0]) == np.ndarray
            assert type(validation_data[1]) == np.ndarray
            assert validation_data[0].shape[0] == X.shape[0]
            assert validation_data[1].shape[0] == Y.shape[0]
            assert validation_data[0].shape[1] == validation_data[1].shape[1]

        self.metrics['current_trained_epochs'] = 0
        optimizer.initiate_parameters(self.layers)
        for _ in range(1, epochs + 1):
            self.layers_backup = copy.deepcopy(self.layers)
            mini_batchs = mini_batch_generator(X, Y, batch_size)
            for mini_X, mini_Y in mini_batchs:
                self.fit_minibatch(mini_X, mini_Y, lambda_, optimizer)

            regularization_cost = get_regularization_cost(
                self.layers, m, lambda_)

            train_Y_hat = self.feed_forward(X)
            train_cost = self.cost_function(Y, train_Y_hat)
            train_cost += regularization_cost
            train_prediction = self.predict_function(train_Y_hat)
            train_accuracy = self.accuracy_func(train_prediction, Y)
            self.metrics['costs'].append(train_cost)
            self.metrics['accuracies'].append(train_accuracy)

            if validation_data:
                validation_X = validation_data[0]
                validation_Y = validation_data[1]
                validation_Y_hat = self.feed_forward(validation_X)
                validation_cost = self.cost_function(validation_Y,
                                                     validation_Y_hat)
                validation_cost += regularization_cost
                validation_prediction = self.predict_function(validation_Y_hat)
                validation_accuracy = self.accuracy_func(
                    validation_prediction, validation_Y)
                self.metrics['validation_costs'].append(validation_cost)
                self.metrics['validation_accuracies'].append(
                    validation_accuracy)

            self.metrics['total_trained_epochs'] += 1
            self.metrics['current_trained_epochs'] += 1

            if metrics_printer:
                metrics_printer(self.metrics)

            optimizer.update_optimizer(self.metrics)
        return self.metrics