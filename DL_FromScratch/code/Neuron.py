import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



input_size = 2
output_size = 3

layer1 = DenseLayer(input_size, output_size)
activation1 = ReLU()


X = np.array([[1, 2], [-1, -2]])

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)

print(layer1.weights)
print("Output of the first layer with ReLU activation:")
print(activation1.output)
