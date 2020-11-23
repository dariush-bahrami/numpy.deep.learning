import numpy as np
import matplotlib.pyplot as plt
from activation_functions import *


def initialize(layers: list):
    result = []
    previous_layer = layers[0]
    for layer in layers[1:]:
        shape = (layer.units_number, previous_layer.units_number)
        # On weight initialization in deep neural networks
        scale_dict = {
            'relu': np.sqrt(2 / shape[1]),
            'tanh': np.sqrt(1 / shape[1]),
            'sigmoid': np.sqrt((3.6**2) / shape[1]),
            'softmax': np.sqrt(2 / shape[1]),
            'identity': np.sqrt(2 / shape[1])
        }
        func_name = layer.activation_function.__name__
        layer.W = np.random.randn(shape[0], shape[1]) * scale_dict[func_name]

        layer.b = np.zeros((shape[0], 1))
        result.append(layer)
        previous_layer = layer
    return result


def mini_batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int):
    assert x.shape[1] == y.shape[1]
    m = x.shape[1]
    random_indice = np.random.permutation(m)
    shuffled_x = x[:, random_indice]
    shuffled_y = y[:, random_indice]

    div = divmod(m, batch_size)
    for i in range(div[0]):
        x_mini_batch = shuffled_x[:, i * batch_size:(i + 1) * batch_size]
        x_mini_batch = x_mini_batch.reshape(x.shape[0], batch_size)

        y_mini_batch = shuffled_y[:, i * batch_size:(i + 1) * batch_size]
        y_mini_batch = y_mini_batch.reshape(y.shape[0], batch_size)

        yield x_mini_batch, y_mini_batch

    if div[1]:
        x_mini_batch = shuffled_x[:, div[0] * batch_size:]
        x_mini_batch = x_mini_batch.reshape(x.shape[0], div[1])

        y_mini_batch = shuffled_y[:, div[0] * batch_size:]
        y_mini_batch = y_mini_batch.reshape(y.shape[0], div[1])

        yield x_mini_batch, y_mini_batch


def get_predict_function(apply_softmax=True):
    def predict(model_output):
        prediction = np.where(model_output > 0.5, 1, 0)
        return prediction

    def softmax_predict(Z):
        A = softmax(Z)
        return predict(A)

    if apply_softmax:
        return softmax_predict
    else:
        return predict


def get_accuracy_function(binary=False):
    def binary_accuracy(prediction, expected):
        assert prediction.shape == expected.shape
        #     assert prediction.shape[0] == 1
        m = expected.shape[1]
        accuracy = np.sum(prediction == expected) / m
        return accuracy

    def categorical_accuracy(prediction, expected):
        assert prediction.shape == expected.shape
        assert prediction.shape[0] != 1
        m = expected.shape[1]
        accuracy = np.sum(
            np.argmax(prediction, axis=0) == np.argmax(expected, axis=0)) / m
        return accuracy

    if binary:
        return binary_accuracy
    else:
        return categorical_accuracy


def print_metrics(interval=100):
    def result_function(metrics):
        epoch = metrics['total_trained_epochs']
        if (epoch == 1) or (epoch % interval == 0):
            cost = metrics['costs'][-1]
            accuracy = metrics['accuracies'][-1] * 100
            message_parts = [
                f'Epoch #{epoch:0>4}', f'Cost: {cost:.4f}',
                f'Accuracy: {accuracy:.2f}%'
            ]

            if metrics['validation_costs']:
                validation_cost = metrics['validation_costs'][-1]
                validation_accuracy = metrics['validation_accuracies'][-1] * 100
                message_parts.append(f'Validation Cost: {validation_cost:.4f}')
                message_parts.append(
                    f'Validation Accuracy: {validation_accuracy:.2f}%')

            print(' | '.join(message_parts))

    return result_function


def plot_metrics(metrics: dict, interval=100):
    cost = metrics['costs']
    accuracy = metrics['accuracies']
    validation_cost = metrics['validation_costs']
    validation_accuracies = metrics['validation_accuracies']

    fig, axes = plt.subplots(2, 1, constrained_layout=True)

    axes[0].plot(cost[::interval], label='Training Cost')
    axes[0].plot(validation_cost[::interval], label='Validation Cost')
    axes[0].set_ylabel('epoch')
    axes[0].set_ylabel('Cost')
    axes[0].legend()

    axes[1].plot(accuracy[::interval], label='Training Accuracy')
    axes[1].plot(validation_accuracies[::interval],
                 label='Validation Accuracy')
    axes[1].set_ylabel('epoch')
    axes[1].set_ylabel(f'Accuracy')
    axes[1].legend()

    # fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.set_size_inches(12, 9)
    plt.show()
    return fig

def one_hot_encode(Y, C):
    assert Y.ndim == 1
    m = Y.shape[0]
    shape = (C, m)
    result = np.zeros(shape)
    result[Y,np.arange(m)] = 1
    return result

def get_dataset(dataset_path):
    data = np.load(dataset_path)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    return X_train, Y_train, X_test, Y_test