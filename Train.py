from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

# Load MNIST dataset with 80% for training and 20% for validation
train_ds, val_ds = tfds.load('mnist', split=['train[:80%]', 'train[80%:100%]'], shuffle_files=True)

def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize to [0, 1]
    label = data['label']
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize to [0, 1]
    label = data['label']
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

train_ds = train_ds.map(augment)
val_ds = val_ds.map(augment)

# Define a simple CNN model
def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'), layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))  # Final layer with softmax activation
    return model

# Create an instance of the model
model = get_model()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Save the model
model.save('model.keras')
