from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load MNIST dataset with 80% for training and 20% for validation


def get_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Final layer with sigmoid activation
    return model

