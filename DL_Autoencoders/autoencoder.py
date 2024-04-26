import os
# Disable oneDNN custom operations to prevent the TensorFlow warning about floating-point round-off errors.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras.models import Model
from convertImages import split_images, split_labels, merge_images, merge_labels
from stacked_mnist_tf import DataMode, StackedMNISTData


class Autoencoder(Model):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoded_size = 16
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(64, (3, 3), activation='relu', strides=1), #8 32 32
            layers.Conv2D(32, (3, 3), activation='relu', strides=1), #8 32 32
            layers.Conv2D(16, (3, 3), activation='relu', strides=1), #8 32 32
            layers.Conv2D(12, (2, 2), activation='relu', strides=1), #8 32 16
            layers.Conv2D(8, (2, 2), activation='relu', strides=1), #8 32
            layers.Conv2D(8, (2, 2), activation='relu', strides=1), #8 16
            layers.Flatten(),
            layers.Dense(1024, activation='relu'), #1024 28 * 28
            layers.Dense(256, activation='relu'), #256 256
            layers.Dense(32, activation='relu'), #32 10
            layers.Dense(self.encoded_size, activation='sigmoid') # 4-8 -
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(self.encoded_size, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(18*18, activation='relu'),
            layers.Reshape((18, 18, 1)),
            layers.Conv2DTranspose(8, kernel_size=2, strides=1, activation='relu'),
            layers.Conv2DTranspose(12, kernel_size=2, strides=1, activation='relu'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu'),
            layers.Conv2DTranspose(1, kernel_size=3, strides=1, activation='sigmoid')
        ])
        """self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(8, (3, 3), activation='relu', strides=1), #8 32 32
            layers.Conv2D(8, (2, 2), activation='relu', strides=1), #8 32 16
            layers.Conv2D(8, (2, 2), activation='relu', strides=1), #8 32
            layers.Conv2D(8, (2, 2), activation='relu', strides=1), #8 16
            layers.Flatten(),
            layers.Dense(1024, activation='relu'), #1024 28 * 28
            layers.Dense(256, activation='relu'), #256 256
            layers.Dense(32, activation='relu'), #32 10
            layers.Dense(8, activation='relu') # 4-8 -
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(23*23, activation='relu'),
            layers.Reshape((23, 23, 1)),
            layers.Conv2DTranspose(8, kernel_size=2, strides=1, activation='relu'),
            layers.Conv2DTranspose(8, kernel_size=2, strides=1, activation='relu'),
            layers.Conv2DTranspose(8, kernel_size=2, strides=1, activation='relu'),
            layers.Conv2DTranspose(1, kernel_size=3, strides=1, activation='sigmoid')
        ])"""

        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    def generate_images(self, num_samples=8):
        # Sample from a standard normal distribution
        # z = np.random.randn(num_samples, 5).reshape(num_samples, 28, 28, 1)
        z = np.random.randn(num_samples, self.encoded_size).astype('float32')

        # Generate images
        generated_images = self.decoder(z)
        return generated_images.numpy()
    
    def predictRGB(self, inputs):
        grayScaleInputs = split_images(inputs)
        greyScalePredictions = self.call(grayScaleInputs).numpy()
        RGBPredictions = merge_images(greyScalePredictions)
        return RGBPredictions
    
    def generateRGB(self, num_samples=8):
        grayScalePredictions = self.generate_images(num_samples*3)
        RGBPredictions = merge_images(grayScalePredictions)
        return RGBPredictions

def plot_comparisons(original_imgs, reconstructed_imgs=None):
    n = original_imgs.shape[0]  # Assuming original_imgs is a numpy array of shape (n, 28, 28, 1)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_imgs[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_imgs[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=1024*100)
    imgTest, clsTest = gen.get_random_batch(batch_size=8)
    # gen.plot_example(images=imgTest, labels=clsTest)

    img, cls = gen.get_full_data_set(training=True)

    #for img, cls in gen.batch_generator(training=False, batch_size=2048):
    #    print(f"Batch has size: Images: {img.shape}; Labels {cls.shape}")
    
    # Create an instance of the autoencoder
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Stopping at a loss under 0.2
    # early_stopping = EarlyStopping(monitor='loss', patience=1, verbose=1, mode='min', baseline=0.2)
    
    model_checkpoint = ModelCheckpoint("C:/Projects/public/DL_Autoencoders/models/AE_MONO_BINARY_MISSING.keras",
                                       monitor='loss', save_best_only=True, verbose=0, mode='min')
    while True:
        autoencoder.fit(img, img, epochs=20, batch_size=512, shuffle=True, callbacks=[model_checkpoint], validation_split=0.1)

        reconstructed_imgs = autoencoder.predict(imgTest)
        plot_comparisons(imgTest, reconstructed_imgs)

        print(f"reconstructed_imgs.shape {reconstructed_imgs.shape}")
        generated_imgs = autoencoder.generate_images()
        print(f"generated_imgs.shape {generated_imgs.shape}")
        plot_comparisons(generated_imgs, reconstructed_imgs)