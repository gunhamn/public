import numpy as np
import jax.numpy as jnp
import jax
import random
import matplotlib.pyplot as plt
from jax import grad, jit
from plant import Plant
from controller import Controller
from controllerJax import ControllerJax
from controllerNN import ControllerNN, sigmoid, tanh, relu
from plantBathtub import plantBathtub

class Consys:
    def __init__(self, epochs, timesteps, controller, plant, visualise=False, learning_rate=0.1, D_range=[-0.01, 0.01]):
        
        self.epochs = epochs
        self.timesteps = timesteps
        self.controller = controller
        self.plant = plant
        self.visualise = visualise
        self.learning_rate = learning_rate
        self.D_range = D_range
        self.key = jax.random.PRNGKey(random.randint(0, 10000))  # 'seed' is an integer
        self.mse_history = []  # List to store MSE of each epoch
        self.parameter_history = []  # List to store controller parameters of each epoch

    def run_system(self):
        print(f"Running {self.epochs} epochs.")

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.run_epoch(self.controller.parameters, self.plant, self.controller, visualise=True)
            
            # (e) Compute the gradients: ∂(MSE)/∂Ω using Jax
            gradient_fn = jax.value_and_grad(self.run_epoch, argnums=0)
            epoch_mse, gradients = gradient_fn(self.controller.parameters, self.plant, self.controller, visualise=False)
            # print(f"Gradients: {[f'{float(array):.7f}' for array in gradients]}")

            # (f) Update Ω (parameters) based on the gradients.
            if type(self.controller) == ControllerNN:
                self.controller.parameters = [(weight - self.learning_rate * weight_grad, bias - self.learning_rate * bias_grad)
                     for (weight, bias), (weight_grad, bias_grad) in zip(self.controller.parameters, gradients)]
                """
                self.controller.parameters =[tuple(np.subtract(param, self.learning_rate * grad) 
                    for param, grad in zip(param_tuple, grad_tuple))
                        for param_tuple, grad_tuple in zip(self.controller.parameters, gradients)]
                """
            else:
                self.controller.parameters = [param - self.learning_rate * grad for param, grad in zip(self.controller.parameters, gradients)]
            # print grandients 
            # print(f"Gradients: {[f'{float(array):.7f}' for array in gradients]}")(weight_grad, bias_grad) in zip(self.controller.parameters, gradients)
            print(f"Gradients: {[(weight_grad, bias_grad) for (weight_grad, bias_grad) in gradients]}")
            #print(f"Parameters: {[f'{float(array):.7f}' for array in self.controller.parameters]}")
            
            self.mse_history.append(epoch_mse)
            self.parameter_history.append(self.controller.parameters)

    def run_epoch(self, parameters, plant, controller, visualise):
        
        self.key, subkey = jax.random.split(self.key)
        D_time_series = jax.random.uniform(subkey, (self.timesteps,), minval=self.D_range[0], maxval=self.D_range[1])
        
        error_history = jnp.zeros(len(D_time_series), dtype=float)
        control_signal = 0
        state = plant.state

        # For each timestep
        for timestep in range(len(D_time_series)):   
            if visualise:
                print(f"Timestep: {timestep}, Plant state: {state}")
            
            # Update the plant
            state = plant.update(   
                state = state,
                input = control_signal,
                disturbance = D_time_series[timestep],
                parameters = plant.parameters)
            
            if visualise:
                print(f"Timestep: {timestep}, Plant state: {state}, control_signal: {control_signal}")
            
            # Save the error (E) for this timestep in an error history
            error = plant.goal_state - state
            error_history = error_history.at[timestep].set(error)

            # Update the controller
            control_signal = controller.update(
                parameters = parameters,
                error = error_history[timestep],
                prev_error = error_history[timestep - 1],
                integral = jnp.sum(error_history))
            
            
            
        # (d) Compute MSE over the error history.
        MSE = jnp.mean(jnp.square(error_history))
        if visualise:
            print(f"MSE: {MSE}")

        return MSE
        

    def plot_Jax_results(self):
        # Plot MSE over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.mse_history, label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE over Epochs')
        plt.legend()
        plt.show()

        # Plot control parameters over epochs
        # Assuming parameters are a list of lists (each sublist contains parameters for an epoch)
        parameter_array = np.array(self.parameter_history)
        plt.figure(figsize=(10, 5))
        for i in range(parameter_array.shape[1]):  # Loop over parameters
            plt.plot(parameter_array[:, i], label=f'Parameter {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Value')
        plt.title('Control Parameters over Epochs')
        plt.legend()
        plt.show()
    
    def plot_NN_results(self):
        # Plot MSE over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.mse_history, label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE over Epochs')
        plt.legend()
        plt.show()
        
        epochs = len(self.parameter_history)
        layers = len(self.parameter_history[0])  # Number of layers
                
        # Plot evolution of weights and biases for each layer
        for layer in range(layers):
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'Layer {layer+1} Parameters Evolution')

            # Assuming the first item in each tuple is the weight matrix, and the second is the bias vector
            weight_shapes = self.parameter_history[0][layer][0].shape
            bias_shapes = self.parameter_history[0][layer][1].shape

            # Initialize arrays to hold parameter evolution data
            weight_data = np.zeros((epochs, np.prod(weight_shapes)))
            bias_data = np.zeros((epochs, bias_shapes[0]))

            for epoch, params in enumerate(self.parameter_history):
                weights, biases = params[layer]
                weight_data[epoch, :] = weights.flatten()
                bias_data[epoch, :] = biases.flatten()

            # Plotting weights
            for i in range(weight_data.shape[1]):
                ax[0].plot(range(epochs), weight_data[:, i], label=f'Weight {i+1}' if i < 10 else None)  # Limiting legend items

            # Plotting biases
            for i in range(bias_data.shape[1]):
                ax[1].plot(range(epochs), bias_data[:, i], label=f'Bias {i+1}' if i < 10 else None)  # Limiting legend items

            ax[0].set_ylabel('Weights')
            ax[1].set_ylabel('Biases')
            ax[1].set_xlabel('Epoch')
            ax[0].legend()
            ax[1].legend()
            plt.show()


if __name__ == "__main__":
    """
    controller = ControllerJax(kp=0.1, kd=0.10, ki=0.05)
    plant = plantBathtub(cross_sectional_area=1, drain_area=0.01, initial_height=1)
    consys = Consys(20, 20, controller, plant, learning_rate=0.01, D_range=[-0.01, 0.01])

    consys.run_system()
    consys.plot_Jax_results()
    """
    hidden_layers = [4]
    activation_functions = [sigmoid]
    controller = ControllerNN(hidden_layers, activation_functions, range_init=[-0.1, 0.1])
    plant = plantBathtub(cross_sectional_area=1, drain_area=0.01, initial_height=1)
    consys = Consys(5, 10, controller, plant, learning_rate=0.01, D_range=[-0.01, 0.01])

    consys.run_system()
    consys.plot_NN_results()
    
    

    