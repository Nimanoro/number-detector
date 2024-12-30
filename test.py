import matplotlib.pyplot as plt
from tensorflow.keras import models
import tensorflow_datasets as tfds
import tensorflow as tf

# Load MNIST test dataset
test_ds = tfds.load('mnist', split='test', shuffle_files=True)

# Normalize images and extract 'image' and 'label'
def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize to [0, 1]
    label = data['label']
    return image, label

test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Load the trained model
model = models.load_model('model.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

# Visualize predictions on the first 10 test samples
fig, axes = plt.subplots(1, 10, figsize=(20, 5))
for i, (images, labels) in enumerate(test_ds.take(1)):  # Take one batch of test samples
    for j in range(10):  # Plot the first 10 samples
        ax = axes[j]
        ax.imshow(images[j].numpy().squeeze(), cmap='gray')
        ax.axis('off')
        prediction = model.predict(images[j:j+1]).argmax()  # Predict for the single image
        ax.set_title(f"P: {prediction}\nT: {labels[j].numpy()}")
plt.tight_layout()
plt.show()
