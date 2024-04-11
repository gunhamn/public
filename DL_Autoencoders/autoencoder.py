import os
# Disable oneDNN custom operations to prevent the TensorFlow warning about floating-point round-off errors.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers
from keras.models import Model

from stacked_mnist_tf import DataMode, StackedMNISTData


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(18, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(18, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        print(f"encoded.shape {encoded.shape}")
        decoded = self.decoder(encoded)
        return decoded
    
    def generate_images(self, num_samples=8):
        # Sample from a standard normal distribution
        # z = np.random.randn(num_samples, 5).reshape(num_samples, 28, 28, 1)
        z = np.random.randn(num_samples, 7, 7, 2).astype('float32')

        # Generate images
        generated_images = self.decoder(z)
        return generated_images.numpy()
    

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
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=9)
    imgTest, clsTest = gen.get_random_batch(batch_size=8)
    gen.plot_example(images=imgTest, labels=clsTest)

    for img, cls in gen.batch_generator(training=False, batch_size=2048):
        print(f"Batch has size: Images: {img.shape}; Labels {cls.shape}")
    
    # Create an instance of the autoencoder
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(img, img, epochs=250, batch_size=256, shuffle=True)

    reconstructed_imgs = autoencoder.predict(imgTest)
    plot_comparisons(imgTest, reconstructed_imgs)

    # Generate 8 random noise images
    random_noise = np.random.rand(8, 28*28).reshape(8, 28, 28, 1)

    print(f"reconstructed_imgs.shape {reconstructed_imgs.shape}")
    generated_imgs = autoencoder.generate_images()
    print(f"generated_imgs.shape {generated_imgs.shape}")
    plot_comparisons(generated_imgs, reconstructed_imgs)

    

