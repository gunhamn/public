
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from plant import Plant


class plantBathtub(Plant):
    def __init__(self, cross_sectional_area, drain_area, initial_height, gravitational_constant=9.8):
        """
        Initializes the Bathtub object as a subclass of Plant.
        """
        super().__init__(
            parameters = jnp.array([cross_sectional_area, drain_area, gravitational_constant]),
            state = initial_height,
            goal_state = initial_height)


    def update(self, state, input, disturbance, parameters):
        """
        Overrides the update method of Plant. Updates the height of the water in the bathtub.

        Returns:
        float: Updated height of water in the bathtub.
        """
        A, C, g = parameters
        H = state

        V = jnp.sqrt(2 * g * H)  # Velocity of water exiting through the drain
        Q = V * C  # Flow rate of exiting water
        delta_B = (input + disturbance - Q)  # Change in bathtub volume
        delta_H = delta_B / A  # Change in water height
        new_state = state + delta_H # New water height

        return jnp.maximum(new_state, 0)  # Water height cannot be negative
    
    def updateDerivative(self, state, input, disturbance, parameters):
        return grad(self.update, argnums=0)(state, input, disturbance, parameters)
