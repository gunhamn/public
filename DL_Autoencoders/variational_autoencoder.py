import os
# Disable oneDNN custom operations to prevent the TensorFlow warning about floating-point round-off errors.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as tfk
from tf_keras import Model
import tensorflow_probability as tfp
from keras.callbacks import ModelCheckpoint

tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions



from stacked_mnist_tf import DataMode, StackedMNISTData


class VariationalAutoencoder(Model):
    def __init__(self, *args, **kwargs):
        super(VariationalAutoencoder, self).__init__()
        self.inputShape = (28, 28, 1)
        self.encoded_size = 16
        self.base_depth = 32

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.encoded_size), scale=1),
                                reinterpreted_batch_ndims=1)

        self.encoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=self.inputShape),
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5), # Normalizes the input data
            tfkl.Conv2D(self.base_depth, 5, strides=1,
                        padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(self.base_depth, 5, strides=2,
                        padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(2 * self.base_depth, 5, strides=1,
                        padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(2 * self.base_depth, 5, strides=2,
                        padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(4 * self.encoded_size, 7, strides=1,
                        padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Flatten(),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.encoded_size),
                    activation=None),
            tfpl.MultivariateNormalTriL(
                self.encoded_size,
                activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
        ])

        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=[self.encoded_size]),
            tfkl.Reshape([1, 1, self.encoded_size]),
            tfkl.Conv2DTranspose(2 * self.base_depth, 7, strides=1,
                                padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(2 * self.base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(2 * self.base_depth, 5, strides=2,
                                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(self.base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(self.base_depth, 5, strides=2,
                                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(self.base_depth, 5, strides=1,
                                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(filters=1, kernel_size=5, strides=1,
                        padding='same', activation=None),
            tfkl.Flatten(),
            tfpl.IndependentBernoulli(self.inputShape, tfd.Bernoulli.logits),
        ])
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        print(f"encoded.shape {encoded.shape}")
        decoded = self.decoder(encoded)
        return decoded
    
    def generate_images(self, num_samples=8):
        # Sample from a standard normal distribution
        # z = np.random.randn(num_samples, 5).reshape(num_samples, 28, 28, 1)
        # z = np.random.randn(num_samples, self.encoded_size).astype('float32')
        z = self.prior.sample(num_samples)
        # Generate images
        generated_images = self.decoder(z)
        return generated_images.sample().numpy()
    

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
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=1024)
    img, cls = gen.get_full_data_set(training=True)
    imgTest, clsTest = gen.get_random_batch(batch_size=8)
    # Create an instance of the autoencoder
    autoencoder = VariationalAutoencoder()
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    autoencoder.compile(optimizer='adam', loss=negative_log_likelihood)
        # loss = negative_log_likehood
        # learning_rate = 0.001

    #model_checkpoint = ModelCheckpoint("C:/Projects/public/DL_Autoencoders/models/autoencoder.keras",
    #                                   monitor='loss', save_best_only=True, verbose=0, mode='min')
    while True:
        #autoencoder.fit(img, img, epochs=50, batch_size=256*4, shuffle=True, callbacks=[model_checkpoint])
        autoencoder.fit(img, img, epochs=10, batch_size=256, shuffle=True)
        # validation_data = ____

        reconstructed_imgs = autoencoder.predict(imgTest)
        plot_comparisons(imgTest, reconstructed_imgs)

        # Generate 8 random noise images
        # random_noise = np.random.rand(8, 28*28).reshape(8, 28, 28, 1)

        print(f"reconstructed_imgs.shape {reconstructed_imgs.shape}")
        generated_imgs = autoencoder.generate_images()
        print(f"generated_imgs.shape {generated_imgs.shape}")
        plot_comparisons(generated_imgs, reconstructed_imgs)
        autoencoder.save("C:/Projects/public/DL_Autoencoders/models/variational_autoencoder.keras")
    

