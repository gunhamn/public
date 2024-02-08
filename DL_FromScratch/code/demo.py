"""
Todo:


c - Number of layers: 0-5 hidden
c - Number of neurons: 1-1000
c - Activation functions: ReLU, Sigmoid, Tanh, Softmax, Linear
c - Loss functions: MSE, Cross-entropy
c - Regularization: L1, L2, none. Rate=0.001
c - Initial w and b for each layer
c - Verbose: network input and output, target values and loss
- Plot loss per minibatch (Train: 70%, Validate: 20%, Test: 10%)

- Data Generation
    - Binary pixels
    - Number of images: n
    - Image dimension: 10 ≤ n ≤ 50
    - At least 4 different classes of objects (approximately the same number of images of each class)
    - They should not always be centered
    - returns 70training, 20validation, 10test
    - Height and width of the objects: 10 ≤ n ≤ 50
    - Flattened:T/F
    - Noise parameter


- Visualise 10 images
- 3 Configuration files:
    1. A configuration file for a network with at least 2 hidden layers that runs on a dataset of at least 500
    training cases and that has previously shown some learning progress (i.e. loss clearly declines over time
    / minibatches). All parameters of this network should be tuned to those that have worked well in the
    past.

    2. A configuration file for a network with no hidden layers that runs on the same 500-item training set
    as above. This network may or may not exhibit learning progress.
    
    3. A configuration file for a network with at least 5 hidden layers that runs on a dataset of at least 100
    training cases for a minimum of 10 passes through the entire training set.


Config example:

GLOBALS
loss: cross_entropy lrate: 0.1 wreg: 0.001 wrt: L2
LAYERS
input: 20
size: 100 act: relu wr: (-0.1 0.1) lrate: 0.01
size: 5 act: relu wr: glorot br: (0 1)
type: softmax
"""