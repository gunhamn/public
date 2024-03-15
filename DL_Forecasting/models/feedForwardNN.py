from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

class feedForwardNN:
    def __init__(self):
        self.model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=[6]),
            layers.Dense(10, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
    
    def fit(self, train_x, train_y, val_x, val_y, epochs=5):
        self.history = self.model.fit(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y))
        return self.history
    
    def evaluate(self, val_x, val_y):
        val_loss = self.model.evaluate(val_x, val_y)
        print(f"Validation loss: {val_loss}")
        return val_loss
    
    def plot_history(self, savePlot=None):
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        if savePlot is not None:
            plt.savefig(savePlot)  # Save the plot to the specified file
        plt.show()

    def save(self, filename='my_model.keras'):
        self.model.save(filename)
