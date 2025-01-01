from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from Generator import get_generator
from Discriminator import get_discriminator
import os
import time

# Load MNIST dataset with 80% for training and 20% for validation
train_ds, val_ds = tfds.load('mnist', split=['train[:80%]', 'train[80%:100%]'], shuffle_files=True)

def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize to [0, 1]
    label = data['label']
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
generator = get_generator()
discriminator = get_discriminator()

# Define the loss functions for the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training loop
import os
import time

# Define the checkpoint directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Define the training parameters
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

# Define the training step
@tf.function
def train_step(images):
    noise = tf.random.normal([200, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, 16)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Train the model
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        noise = tf.random.normal([200, noise_dim])
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF
        generate_and_save_images(generator, epoch + 1, noise)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    generator.save('generator.keras')
    discriminator.save('discriminator.keras')

    generate_and_save_images(generator, epochs, noise)

# Train the model
train(train_ds, EPOCHS)
# Save the model
# Save the checkpoint
checkpoint.save(file_prefix = checkpoint_prefix)
# Load the model
