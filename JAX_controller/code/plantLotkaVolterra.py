import jax.numpy as jnp
from plant import Plant


class plantLotkaVolterra(Plant):
    def __init__(self, rabbits=1, foxes=1, goal_state=2, rabbitBirthRate = 0.1, rabbitDeathRatePerFox = 0.02, foxDeathRate = 0.1, foxBirthRatePerRabbit = 0.01):
        """
        Initializes the Cournot simulation as a subclass of Plant.
        """
        super().__init__(
            parameters = jnp.array([rabbitBirthRate, rabbitDeathRatePerFox, foxDeathRate, foxBirthRatePerRabbit]),
            state = jnp.array([rabbits, foxes]),
            goal_state = goal_state)


    def update(self, state, input, disturbance, parameters):
        
        rabbitBirthRate, rabbitDeathRatePerFox, foxDeathRate, foxBirthRatePerRabbit = parameters
        rabbits, foxes = state

        rabbits += input
        foxes += disturbance

        # Update the populations based on Lotka-Volterra equations
        rabbits = jnp.maximum(0, rabbits + (rabbitBirthRate * rabbits - rabbitDeathRatePerFox * rabbits * foxes))
        foxes = jnp.maximum(0, foxes + (foxBirthRatePerRabbit * rabbits * foxes - foxDeathRate * foxes))

        error = self.goal_state - rabbits

        return error, jnp.array([rabbits, foxes])