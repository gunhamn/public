import numpy as np
from activation import get_activation


class Dense:
    def __init__(self, output_size: int, activation: str, input_size: int, init_range=[-0.1, 0.1]):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = get_activation(activation)
        self.init_range = init_range
        self.W = np.random.uniform(self.init_range[0], self.init_range[1], (self.input_size, self.output_size))
        self.b = np.random.uniform(self.init_range[0], self.init_range[1], (self.output_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Compute the linear part of the forward pass
        self.last_z = np.dot(x, self.W) + self.b
        # Apply the activation function
        return self.activation.func(self.last_z)
    
    def backward(self, input, grad_output):
        # Compute the gradient of the activation function
        grad_activation = self.activation.grad(self.last_z)
        
        # Apply the chain rule to get the gradient of the loss with respect to the z (pre-activation)
        grad_z = grad_output * grad_activation
        
        # Compute gradients with respect to weights and biases
        grad_W = np.dot(input.T, grad_z)
        grad_b = np.sum(grad_z, axis=0)
        
        # Compute gradient with respect to the input of the current layer
        grad_input = np.dot(grad_z, self.W.T)
        
        return grad_input, grad_W, grad_b