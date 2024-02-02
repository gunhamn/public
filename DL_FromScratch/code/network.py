import numpy as np
import layers
import lossFunction

class Sequential:
    def __init__(self, layers, learning_rate=0.1, Verbose=False):
        self.layers = layers
        self.learning_rate = learning_rate
        self.lossFunction = lossFunction.get_lossFunction('mse')
        self.Verbose = Verbose
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def compile():
        pass

    def loss(self, predictions, targets):  
        return self.lossFunction(predictions, targets)

    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def fit(self, x_train, y_train, epochs=1):
        for epochNr in range(epochs):
            print('Epoch:', epochNr)
            for x, y in zip(x_train, y_train):
                # Ensures that x and y are 2D (1, input_size)
                if x.ndim == 1:
                    x = x[np.newaxis, :]  # Add batch dimension if not present
                if y.ndim == 1:
                    y = y[np.newaxis, :]  # Add batch dimension if not present

                # Make predictions of y
                predictions = self.forward(x)

                # Calculate the loss
                loss, lossGrad = self.loss(predictions, y)

                if self.Verbose:
                    print('Input:', x)
                    print('Output:', predictions)
                    print('Target:', y)
                    print('Loss:', loss)

                # Calculate gradients of the weights and biases
                # with respect to loss, these are saved in the layers
                self.backward(lossGrad)

                # Update the weights and biases
                # using the layers update function
                self.update()