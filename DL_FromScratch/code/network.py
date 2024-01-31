import numpy as np
import layers


class Sequential:
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def compile():
        pass

    def loss(self, predictions, targets):  
        pass

    def update(self, gradients):
        pass

    def fit(self, x_train, y_train, epochs=5):
        for _ in range(epochs):
            for x, y in zip(x_train, y_train):

                # Make predictions of y
                predictions = self.forward(x)

                # Calculate the loss
                loss = self.loss(predictions, y)

                # Calculate gradients of the weights and biases
                # with respect to loss
                gradients = self.backward(loss)

                # Update the weights and biases
                self.update(gradients)



if __name__ == "__main__":
    # Define the network architecture
    network = Sequential([
        layers.Dense(output_size=64, activation='relu', input_size=100),
        layers.Dense(output_size=3, activation='softmax', input_size=64)
    ])

    # Initialize some input data
    # Example: batch of 10 samples, each with 100 features
    X = np.random.randn(10, 100)

    # Perform a forward pass through the network
    output = network.forward(X)

    print("Network output:", output)