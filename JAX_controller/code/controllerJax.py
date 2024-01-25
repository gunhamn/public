import jax.numpy as jnp
from controller import Controller

class ControllerJax:
    def __init__(self, kp, kd, ki):
        """
        kp # Proportional gain.
        kd # Derivative gain.
        ki # Integral gain.
        """
        self.parameters = jnp.array([kp, kd, ki])

    def update(self, parameters, error, prev_error, integral):

        kp, kd, ki = parameters
        derivative = error - prev_error

        # U = kp*E + kd*(dE/dt) + ki*integral( E)
        output_signal = kp*error + kd*derivative + ki*integral
        
        return output_signal
