class Controller:
    def __init__(self, kp, ki, kd):
        """
        Initializes the Controller object.

        Parameters:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, timestamp):
        """
        Updates the controller's output based on the error.

        Parameters:
        error (float): The error signal (goal_state - current_state).
        dt (float): Time interval.

        Returns:
        float: The control input.
        """
        self.integral += error * (timestamp+1)
        derivative = (error - self.previous_error) / (timestamp+1)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output
