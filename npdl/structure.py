from activation_functions import *

activation_functions_dict = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, None),
    'identity': (identity, identity_derivative)
}


class LayerNamespace(object):
    pass


class ModelStructure(object):
    def __init__(self):
        self.__layers = []

    def add_input_layer(self, input_features: int):
        assert len(self.__layers) == 0, 'Add input layer before other layers'
        assert type(input_features) == int, 'Enter integer number'
        layer_0 = LayerNamespace()
        layer_0.units_number = input_features
        self.__layers.append(layer_0)

    def add_layer(self,
                  units_number: int,
                  activation_function_name='identity',
                  keep_prob=1):
        assert type(units_number) == int, 'Enter integer number'
        assert len(self.__layers) > 0, 'First add input layer'

        try:
            layer = LayerNamespace()
            valid_key = activation_function_name.lower()
            layer.activation_function = activation_functions_dict[valid_key][0]
            layer.activation_function_derivative = activation_functions_dict[
                valid_key][1]
            layer.units_number = units_number
            layer.keep_prob = keep_prob
            self.__layers.append(layer)
        except KeyError as error:
            message = 'Valid activation functions: '
            message += ', '.join(activation_functions_dict.keys())
            raise KeyError(f'{error} ({message})')

        return True

    @property
    def layers(self):
        assert len(self.__layers) > 1, 'At least insert one layer'
        return self.__layers

    def summary(self):
        print(f'Input Features: {self.layers[0].units_number}')
        total_parameters = 0
        pre_units = self.layers[0].units_number
        for i, layer in enumerate(self.layers[1:]):
            weights = layer.units_number * pre_units
            biases = layer.units_number
            message_parts = [
                f'Layer #{i+1}:', f'{layer.units_number}',
                f'{layer.activation_function.__name__}',
                f'units with {weights} weights and', f'{biases} biases'
            ]
            print(' '.join(message_parts))
            pre_units = layer.units_number
            total_parameters += (weights + biases)
        print(f'Total Parameters: {total_parameters}')