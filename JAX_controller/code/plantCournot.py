import jax.numpy as jnp
from plant import Plant


class plantCournot(Plant):
    def __init__(self, q1 = 0.5, q2 = 0.5, pMax = 2, goal_state = 1, marginalCost = 0.1):
        """
        Initializes the Cournot simulation as a subclass of Plant.
        """
        super().__init__(
            parameters = jnp.array([pMax, marginalCost]),
            state = jnp.array([q1, q2]),
            goal_state = goal_state)


    def update(self, state, input, disturbance, parameters):
        
        q1, q2 = state
        pMax, marginalCost = parameters
        U = input

        # q1 = jnp.maximum(0, jnp.minimum(1, q1 + U))
        # q2 = jnp.maximum(0, jnp.minimum(1, q2 + disturbance))
        q1 = q1 + U
        q2 = q2 + disturbance
        
        price = pMax - (q1 + q2)
        profit = q1 * (price - marginalCost)
        error = self.goal_state - profit

        q1 = jnp.maximum(0, jnp.minimum(1, q1))
        q2 = jnp.maximum(0, jnp.minimum(1, q2))

        return error, jnp.array([q1, q2]),