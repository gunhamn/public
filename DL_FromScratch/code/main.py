"""
# pip install tensorflow

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.shape)
"""

# import os
# os.chdir('path/to/directory')

import sys; print(sys.executable)


import pandas as pd

filename = 'red-wine.csv'

df = pd.read_csv(filename)

print(df.head())
