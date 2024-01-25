class Plant:
    def __init__(self, parameters=None, state = None, goal_state=None):
        """
        Initializes a Plant object.
        
        Parameters:
        parameters (list, optional): List of parameters for the plant.
        inputs (list, optional): List of input values for the plant.
        goal_state (any, optional): Desired goal state for the plant.
        """
        self.parameters = parameters if parameters is not None else []
        self.inputs = []
        self.goal_state = goal_state
        self.state = state

    def update(self, state, input, disturbance, parameters):
        """
        Update method to be overridden in subclasses.
        
        Returns:
        any: Updated states of the plant.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
