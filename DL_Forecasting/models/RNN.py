import tensorflow as tf
from tensorflow import keras
from keras import Model, layers, regularizers
import matplotlib.pyplot as plt

class RNN(Model):
    def __init__(self):
        self.model = keras.Sequential([
            layers.LSTM(50, activation='tanh', input_shape=(24, 6),
                                return_sequences=True,  # Adjust based on whether you want to return sequences
                                dropout=0.2, recurrent_dropout=0.1,
                                kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dense(10, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def fit(self, train_x, train_y, val_x, val_y, epochs=5):
        self.history = self.model.fit(train_x, train_y, epochs=epochs, validation_data=(val_x, val_y))
        return self.history
    
    def evaluate(self, test_x, test_y):
        test_loss = self.model.evaluate(test_x, test_y)
        print(f"Test loss: {test_loss}")
        return test_loss
    
    def plot_history(self, savePlot=None):
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        if savePlot is not None:
            plt.savefig(savePlot)
        plt.show()

    def save(self, filename='my_custom_rnn_model'):
        self.model.save(filename)  # Save the internal Keras model

