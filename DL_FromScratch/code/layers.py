import numpy as np
from activation import get_activation


class Dense:
    def __init__(self, output_size: int, activation: str, input_size: int, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1]):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = get_activation(activation)
        self.W = np.random.uniform(w_range[0], w_range[1], (self.input_size, self.output_size))
        self.b = np.random.uniform(b_range[0], b_range[1], (self.output_size))
        self.last_input = None
        self.last_z = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Save the input to use it in the backward pass

        self.last_input = x

        # Compute the linear part of the forward pass
        self.last_z = np.dot(x, self.W) + self.b

        # Apply the activation function
        return self.activation.func(self.last_z)
    
    def backward(self, grad_output):
        # Compute the gradient of the activation function
        grad_activation = self.activation.grad(self.last_z)
        
        # Apply the chain rule to get the gradient of the loss with respect to the z (pre-activation)
        grad_z = np.multiply(grad_output, grad_activation)
        
        # Compute gradients with respect to weights and biases
        self.grad_W = np.dot(self.last_input.T, grad_z)
        self.grad_b = np.sum(grad_z, axis=0)
        
        # Compute gradient with respect to the input of the current layer
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input
    
    def update(self, learning_rate, regularization, regLambda):
        self.W -= learning_rate * (self.grad_W + regularization.grad(regLambda, self.W))
        self.b -= learning_rate * (self.grad_b + regularization.grad(regLambda, self.b))