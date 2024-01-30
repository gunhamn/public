import jax
import jax.numpy as jnp
from controller import Controller
import random

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def tanh(x):
    return jnp.tanh(x)

def relu(x):
    return jnp.maximum(0, x)

class ControllerNN:
    def __init__(self, hidden_layers=[5, 5, 5], activation_functions = [sigmoid, tanh, relu], range_init=[-0.1, 0.1]):
        self.input_size = 3 # error, derivate, integral
        self.output_size = 1
        self.activation_functions = activation_functions

        #self.parameters = jnp.array([])
        key = jax.random.PRNGKey(random.randint(0, 10000))

        # append input and output size to first and last postision of hidden_layers

        self.parameters = []

        hidden_layers.insert(0, self.input_size)
        hidden_layers.append(self.output_size)
        for i in range(len(hidden_layers)-1):
            key, weight_key, bias_key = jax.random.split(key, 3)

            # initialize weights and biases for each layer using jax random
            weights = jax.random.uniform(weight_key,
                (hidden_layers[i],
                 hidden_layers[i+1]),
                 minval=range_init[0],
                 maxval=range_init[1])
            biases = jax.random.uniform(bias_key,
                (hidden_layers[i+1],),
                minval=range_init[0],
                maxval=range_init[1])
            
            self.parameters.append((weights, biases))
            
    def forward(self, parameters, x):
        # hidden layers

        #print(f"activation_functions.len: {len(self.activation_functions)}")
        #print("\n".join([f"Layer {i+1} - Weights shape: {w.shape}, Biases shape: {b.shape}" for i, (w, b) in enumerate(parameters)]))
    
        for (w, b), activation in zip(parameters[:-1], self.activation_functions):
            x = jnp.dot(x, w) + b
            x = activation(x)

        # output layer, no activation function
        (w, b) = parameters[-1]
        x = jnp.dot(x, w) + b
        return sigmoid(x)

    def update(self, parameters, error, prev_error, integral):
        derivative = error - prev_error
        x = jnp.array([error, derivative, integral])
        output_signal = self.forward(parameters, x)

        return output_signal[0]
    


    
if __name__ == "__main__":

    # hidden layer controller
    hidden_layers = [4, 4, 4]
    activation_functions = [sigmoid, tanh, relu]
    controller = ControllerNN(hidden_layers, activation_functions)
    print(controller.update(controller.parameters, 1, 2, 3))
    

    # no hidden layers controller
    hidden_layers = []
    activation_functions = []
    controller = ControllerNN(hidden_layers, activation_functions)
    print(controller.update(controller.parameters, 1, 2, 3))
    