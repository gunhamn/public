import numpy as np
import layers
from lossFunction import get_lossFunction
from regularization import get_regularization

class Sequential:
    def __init__(self, layers, learning_rate=0.1, lossFunction='mse', regularization='L1', regLambda=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.lossFunction = get_lossFunction(lossFunction)
        self.regularization = get_regularization(regularization)
        self.regLambda = regLambda
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def compile():
        pass

    def loss(self, predictions, targets):
        loss, lossGrad = self.lossFunction(predictions, targets)

        if self.regLambda != 0:
            # Add regularization to the loss
            for layer in self.layers:
                loss += self.regularization.func(self.regLambda, layer.W)
                loss += self.regularization.func(self.regLambda, layer.b)
  
        return loss, lossGrad

    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate, self.regularization, self.regLambda)
    
    def create_batches(self, x_train, y_train, batchSize = None):
        # Return the datasets as batches,
        # rounded to the nearest even batch size
        if batchSize is None:
            return np.array([x_train]), np.array([y_train])
        else:
            n_batches = -(-x_train.shape[0] // batchSize)  # Ceiling division to get number of batches
            x_batches = np.array_split(x_train, n_batches)
            y_batches = np.array_split(y_train, n_batches)
            return x_batches, y_batches

    def visualize(self, errorHistory):
        import matplotlib.pyplot as plt
        plt.plot(errorHistory)
        plt.xlabel('Batch')
        plt.ylabel('Error')
        plt.show()

    def fit(self, x_train, y_train, epochs=1, batchSize=None, visualize=True, verbose=False, randomize=True):
        errorHistory = []
        for epochNr in range(epochs):
            print('Epoch:', epochNr)

            if randomize:
                # Shuffle the training data
                permutation = np.random.permutation(x_train.shape[0])
                x_train = x_train[permutation]
                y_train = y_train[permutation]
            
            x_trainBatches, y_trainBatches = self.create_batches(x_train, y_train, batchSize=batchSize)
            for x_batch, y_batch in zip(x_trainBatches, y_trainBatches):
                # Make predictions for all samples
                predictions = self.forward(x_batch)
                
                # Calculate the loss for the entire batch
                loss, lossGrad = self.loss(predictions, y_batch)
                
                if verbose:
                    print('Input:', x_batch)
                    print('Output:', predictions)
                    print('Target:', y_batch)
                    print('Loss:', loss)
                
                # Backpropagate the error for the entire batch
                self.backward(lossGrad)
                
                # Update the weights and biases for the entire batch
                self.update()

                print('Loss:', loss)
                errorHistory.append(loss)
        if visualize:
            self.visualize(errorHistory)