"""
Parameters should include:
1. The plant to simulate: bathtub, Cournot competition, your additional model, etc.
2. The controller to use: classic or AI-based
3. Number of layers and number of neurons in each layer of the neural network. Your system should
handle anywhere between 0 and 5 hidden layers.
4. Activation function used for each layer of the neural network. Your system must include at least
Sigmoid, Tanh and RELU.
5. Range of acceptable initial values for each weight and bias in the neural network.
6. Number of training epochs
7. Number of simulation timesteps of the CONSYS per epoch
8. Learning rate for tuning the controller parameters, whether classic PID or neural-net-based.
9. Range of acceptable values for noise / disturbance (D).
10. Cross-sectional area (A) of the bathtub
11. Cross-sectional area (C) of the bathtub’s drain.
12. Initial height (H0) of the bathtub water.
13. The maximum price (pmax) for Cournot competition.
14. The marginal cost (cm) for Cournot competition.
15. At least two parameters for your third plant.


Consys:
- nr training epochs
- nr timesteps per epoch
- ranges for D (disturbance)
- learning rate

Controller:
- type of controller
    - classic
        - ranges for k values
    - neural network
        - layers (0 to 5 hidden layers)
            - layer type (Sigmoid, Tanh and RELU)
            - ranges for initial weights

Plant:
- type of plant
    - Bathtub
        - A (Cross-sectional area of the bathtub)
        - C (Cross-sectional area of the bathtub’s drain)
        - H (Initial height and goal)
    - Cournot
        - Pmax (The maximum price)
        - Cm (The marginal cost)
    - Own
        - Parameter 1
        - Parameter 2


"""