#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

"""
    Represents a training step (feed-forward + backpropagation & weights updating)
    Normalizes the images

    <Params>
        images          = images from the data set
        target          = labels from the data set
        latent_size     = size of the latent vector
"""
@tf.function
def train_step(images, target, latent_size, generator_model, discriminator_model):

    # Get a random noise vector for the Generator
    noise = tf.random.normal([target.shape[0], latent_size])

    # TRAIN DISCRIMINATOR WITH REAL LABELS
    # NETWORK VARIABLES RECORDED THROUGH THE GRADIENT TAPE
    with tf.GradientTape() as discriminator_tape_real:
        generated_images    = generator_model([noise, target], training=True)
        real_images         = discriminator_model([images, target], training=True)
        
