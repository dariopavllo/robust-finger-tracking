import numpy as np

LAYER_TYPES = {
    'Dense': 0,
    'SimpleRNN': 1,
    'LSTM': 2
}

ACTIVATION_TYPES = {
    'relu': 0,
    'linear': 1,
    'sigmoid': 2,
    'tanh': 3,
    'hard_sigmoid': 4
}

def write_tensor(file, tensor):
    np.array(tensor.shape, dtype='int32').tofile(file)
    tensor.tofile(file)
    
def write_integer(file, value):
    np.array([value], dtype='int32').tofile(file)

def save_model(model, filename):
    print('List of layers to be exported:')
    num_layers = 0
    for layer in model.layers:
        if type(layer).__name__ == 'TimeDistributed' or type(layer).__name__ == 'Bidirectional':
            layer = layer.layer # Unwrap
        if type(layer).__name__ in LAYER_TYPES:
                num_layers += 1
                print(num_layers,'-', type(layer).__name__, '- Activation:', layer.activation.__name__)
                if layer.activation.__name__ not in ACTIVATION_TYPES:
                    raise Exception('Unsupported activation function: ' + layer.activation.__name__)

    print('Number of layers:', num_layers)

    file = open(filename, 'wb')
    write_integer(file, num_layers) # Write number of layers
    for layer in model.layers:
        if type(layer).__name__ == 'TimeDistributed' or type(layer).__name__ == 'Bidirectional':
            layer = layer.layer # Unwrap
        layer_name = type(layer).__name__
        if layer_name in LAYER_TYPES:
            write_integer(file, LAYER_TYPES[layer_name])
            write_integer(file, ACTIVATION_TYPES[layer.activation.__name__])

            weights = layer.get_weights()
            for i in range(len(weights)):
                write_tensor(file, weights[i])

    file.close()
    print('Model saved to ' + filename)