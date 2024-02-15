import numpy as np
import layers
from lossFunction import get_lossFunction
from regularization import get_regularization
import matplotlib.pyplot as plt

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
        regLoss = 0
        if self.regLambda != 0:
            # Add regularization to the loss
            for layer in self.layers:
                regLoss += self.regularization.func(self.regLambda, layer.W)
  
        return loss, lossGrad, regLoss

    def update(self):
        for layer in self.layers:
            layer.update(self.learning_rate, self.regularization, self.regLambda)
    
    def accuracy(self, x_test, y_test):
        predictions = np.argmax(self.forward(x_test), axis=1)
        labels = np.argmax(y_test, axis=1)
        correct_predictions = np.sum(predictions == labels)
        total_predictions = len(y_test)
        return correct_predictions / total_predictions

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

    def visualize(self, errorHistory, regLossHistory, valErrorHistory):
        plt.plot(errorHistory, label='Training Error')
        plt.xlabel('Batch')
        plt.ylabel('Error')
        plt.plot(regLossHistory, label='Regularization Loss')
        
        # Check if valErrorHistory is provided and not empty
        if len(valErrorHistory) > 0:
            plt.plot(valErrorHistory, label='Validation Error')
            plt.legend()  # This adds a legend to distinguish the plots
        
        plt.show()

    def fit(self, x_train, y_train, validation_data=None, epochs=1, batchSize=None, visualize=True, verbose=False, randomize=True):
        errorHistory = []
        regLossHistory = []
        valErrorHistory = []
        for epochNr in range(epochs):
            print('Epoch:', epochNr+1)

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
                loss, lossGrad, regLoss = self.loss(predictions, y_batch)
                
                if verbose:
                    print('Input:', x_batch)
                    print('Output:', predictions)
                    print('Target:', y_batch)
                    print('Loss:', loss)
                
                # Backpropagate the error for the entire batch
                self.backward(lossGrad)
                
                # Update the weights and biases for the entire batch
                self.update()
                
                errorHistory.append(loss)
                regLossHistory.append(regLoss)

                if validation_data is not None:
                    val_predictions = self.forward(validation_data[0])
                    val_loss, _, _ = self.loss(val_predictions, validation_data[1])
                    valErrorHistory.append(val_loss)
        print(f'Training set accuracy: {self.accuracy(x_train, y_train)}')
        if validation_data is not None:
            print(f'Validation set accuracy: {self.accuracy(validation_data[0], validation_data[1])}')
        if visualize:
            self.visualize(errorHistory, regLossHistory, valErrorHistory)