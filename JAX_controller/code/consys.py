import numpy as np
import jax.numpy as jnp
import jax
import random
import matplotlib.pyplot as plt
from plant import Plant
from controller import Controller
from controllerJax import ControllerJax
from controllerNN import ControllerNN, sigmoid, tanh, relu
from plantBathtub import plantBathtub
from plantCournot import plantCournot
from plantLotkaVolterra import plantLotkaVolterra

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

            # Printing the timesteps here (visualise=True)
            self.run_epoch(self.controller.parameters, self.plant, self.controller, visualise=True)
            
            # (e) Compute the gradients: ∂(MSE)/∂Ω using Jax
            gradient_fn = jax.value_and_grad(self.run_epoch, argnums=0)
            epoch_mse, gradients = gradient_fn(self.controller.parameters, self.plant, self.controller, visualise=False)

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
            
            self.mse_history.append(epoch_mse)
            self.parameter_history.append(self.controller.parameters)

    def run_epoch(self, parameters, plant, controller, visualise):
        
        # Split the jax random key and initialize the disturbance time series
        self.key, subkey = jax.random.split(self.key)
        D_time_series = jax.random.uniform(subkey, (self.timesteps,), minval=self.D_range[0], maxval=self.D_range[1])
        
        # Initialize an empty error history and set the plant to its initial state
        error_history = jnp.zeros(len(D_time_series), dtype=float)
        control_signal = 0
        state = plant.state

        # For each timestep
        for timestep in range(len(D_time_series)):

            # Update the plant
            error, state = plant.update(   
                state = state,
                input = control_signal,
                disturbance = D_time_series[timestep],
                parameters = plant.parameters)
            
            if visualise:
                print(f"Timestep: {timestep}, Plant state: {state}, control_signal: {control_signal}, error: {error}")
            
            # Save the error (E) for this timestep in an error history
            error_history = error_history.at[timestep].set(error)

            # Update the controller
            control_signal = controller.update(
                parameters = parameters,
                error = error,
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
        parameters = ['kp', 'kd', 'ki']
        for i, param_name in enumerate(parameters):
            plt.plot(parameter_array[:, i], label=f'{param_name}')
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
    consys = Consys(50, 10, controller, plant, learning_rate=0.01, D_range=[-0.01, 0.01])
    consys.run_system()
    consys.plot_Jax_results()
    
    hidden_layers = [3, 3, 3]
    activation_functions = [tanh, sigmoid, relu]
    controller = ControllerNN(hidden_layers, activation_functions, range_init=[-0.1, 0.1])
    plant = plantBathtub(cross_sectional_area=1, drain_area=0.01, initial_height=1)
    consys = Consys(50, 30, controller, plant, learning_rate=0.01, D_range=[-0.01, 0.01])
    consys.run_system()
    consys.plot_NN_results()
    

    controller = ControllerJax(kp=0.1, kd=0.10, ki=0.05)
    plant = plantCournot(q1 = 0.5, q2 = 0.5, pMax = 4, goal_state = 2, marginalCost = 0.2)
    consys = Consys(30, 20, controller, plant, learning_rate=0.01, D_range=[-0.02, 0.02])
    consys.run_system()
    consys.plot_Jax_results()

    """
    hidden_layers = [4]
    activation_functions = [relu]
    controller = ControllerNN(hidden_layers, activation_functions, range_init=[-0.1, 0.1])
    plant = plantCournot(q1 = 0, q2 = 0, pMax = 4, goal_state = 2, marginalCost = 0.2)
    consys = Consys(150, 10, controller, plant, learning_rate=0.05, D_range=[-0.01, 0.01])
    consys.run_system()
    consys.plot_NN_results()
    """
    
    controller = ControllerJax(kp=0.1, kd=0.10, ki=0.05)
    plant = plantLotkaVolterra(rabbits = 1, foxes = 1, goal_state = 2,
                               rabbitBirthRate = 0.1,
                               rabbitDeathRatePerFox = 0.02,
                               foxDeathRate = 0.1,
                               foxBirthRatePerRabbit = 0.01)
    consys = Consys(30, 20, controller, plant, learning_rate=0.01, D_range=[-0.02, 0.02])
    consys.run_system()
    consys.plot_Jax_results()
    

    hidden_layers = [4]
    activation_functions = [relu]
    controller = ControllerNN(hidden_layers, activation_functions, range_init=[-0.1, 0.1])
    plant = plantLotkaVolterra(rabbits = 1, foxes = 1, goal_state = 2,
                               rabbitBirthRate = 0.1,
                               rabbitDeathRatePerFox = 0.02,
                               foxDeathRate = 0.1,
                               foxBirthRatePerRabbit = 0.01)
    consys = Consys(30, 10, controller, plant, learning_rate=0.01, D_range=[-0.02, 0.02])
    consys.run_system()
    consys.plot_NN_results()
    

    controller = ControllerJax(kp=0.1, kd=0.10, ki=0.05)
    plant = plantBathtub(cross_sectional_area=1, drain_area=0.01, initial_height=1)
    consys = Consys(50, 10, controller, plant, learning_rate=0.01, D_range=[-0.01, 0.01])
    consys.run_system()
    consys.plot_Jax_results()

    
    hidden_layers = [4]
    activation_functions = [relu]
    controller = ControllerNN(hidden_layers, activation_functions, range_init=[-0.1, 0.1])
    plant = plantLotkaVolterra(rabbits = 1, foxes = 1, goal_state = 2,
                               rabbitBirthRate = 0.1,
                               rabbitDeathRatePerFox = 0.02,
                               foxDeathRate = 0.1,
                               foxBirthRatePerRabbit = 0.01)
    consys = Consys(30, 10, controller, plant, learning_rate=0.01, D_range=[-0.02, 0.02])
    consys.run_system()
    consys.plot_NN_results()
    """