#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from model.functions.losses import discriminator_loss_function, generator_loss_function
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Imports for argument type hints
from model.classes.networks.discriminator.Discriminator import Discriminator
from model.classes.networks.generator.Generator import Generator

"""
    Fix for Issue #5 (Git)
    Source : https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio
"""
tf.config.run_functions_eagerly(True)

@tf.function
def train_step(images, target, latent_size, discriminator_optimizer, generator_optimizer, learning_rate, generator_model: tf.keras.Model, discriminator_model: tf.keras.Model):
    """
        Represents a training step (feed-forward + backpropagation & weights updating)
        Normalizes the images

        <Params>
            images                      = images from the data set
            target                      = labels from the data set
            latent_size                 = size of the latent vector
            discriminator_optimizer     = optimizer to use for the discriminator training step (eg: 'Adam')
            generator_optimizer         = optimizer to use for the generator training step (eg: 'Adam')
            learning_rate               = learning rate
            generator_model             = configured & initialised generator model
            discriminator_model         = configured & initialised discriminator model
    """
    # Get a random noise vector for the Generator
    noise = tf.random.normal([target.shape[0], latent_size])

    ############ TRAIN DISCRIMINATOR WITH REAL LABELS ############
    with tf.GradientTape() as discriminator_tape_real_labels:

        discriminator_real_output    = discriminator_model([images, target], training=True)
            
        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)
        real_targets    = tf.ones_like(discriminator_real_output)
        loss            = discriminator_loss_function('binary_cross_entropy', real_targets, discriminator_real_output)
        print(f'LOSS (DISCRMINATOR [REAL LABELS]) : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_real_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

    ############ TRAIN DISCRIMINATOR WITH FAKE LABELS ############
    with tf.GradientTape() as discriminator_tape_fake_labels:

        generated_images             = generator_model([target, noise], training=True)
        discriminator_fake_output    = discriminator_model([generated_images, target], training=True)
        
        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)   
        fake_targets    = tf.zeros_like(discriminator_fake_output)
        loss            = discriminator_loss_function('binary_cross_entropy', fake_targets, discriminator_fake_output)
        print(f'LOSS (DISCRMINATOR [FAKE LABELS]) : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_fake_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

    ############ TRAIN GENERATOR WITH MATCHING (REAL) LABELS ############
    with tf.GradientTape() as generator_tape:

        generated_images             = generator_model([target, noise], training=True)
        discriminator_fake_output    = discriminator_model([generated_images, target], training=True)

        ### PLOTTING THE GENERATED IMAGE WITH THE LABEL
        display_label = ""
        if (target[0] == 0):
            display_label = "ROCK"
        elif (target[0] == 1):
            display_label = "PAPER"
        else:
            display_label = "SCISSORS"
        generated_image_item = generated_images[0]
        generated_image_item = 255 * generated_image_item
        generated_image_item = tf.cast(generated_image_item, tf.uint8)
        plt.title(display_label)
        plt.imshow(generated_image_item)
        plt.savefig('generated/trainingSample.png')
        # plt.imshow(first_array)

        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)   
        real_targets    = tf.ones_like(discriminator_fake_output)
        loss            = generator_loss_function('binary_cross_entropy', real_targets, discriminator_fake_output)
        print(f'LOSS (GENERATOR [REAL LABELS]) : {loss}')
    
        # Calculate gradient for discriminator training step with real labels
        gradients       = generator_tape.gradient(loss, generator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate)
        optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))

def get_optimizer(optimizer, learning_rate):
    """
        Gets a tf.keras.optimizers optimizer instance

        <Params>
            optimizer       = string argument for the optimizer to get an instance for (eg: 'Adam')
            learning_rate   = learning rate to configure the optimzier with
    """
    if optimizer == 'Adam':
        return optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    elif optimizer == 'Adamax':
        return optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        