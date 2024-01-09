import numpy as np


# Activation Class
class Activation:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

# Activation functions
def relu_function(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

relu = Activation(relu_function, relu_derivative)


class DenseLayer:
    def __init__(self, input_shape, units, activation=relu):
        self.weights = np.random.randn(input_shape, units) * 0.01
        self.biases = np.zeros((1, units))
        self.activation = activation
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        return self.activation.function(self.z)


class SequentialModel:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

model = SequentialModel(
    [DenseLayer(11, 8, activation=relu),
     DenseLayer(8, 8, activation=relu),
     DenseLayer(8, 1, activation=relu)])

random_array = np.random.randn(1, 11)
print(random_array)

print(model.forward(random_array))