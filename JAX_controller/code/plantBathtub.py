
import jax.numpy as jnp
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
        new_state = jnp.maximum(state + delta_H, 0) # New water height, cannot be negative
        error = self.goal_state - new_state

        return error, new_state