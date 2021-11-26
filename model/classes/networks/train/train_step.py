#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_math_ops import add
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

# Global loss arrays for evaluation purposes
global_d_loss   = []
global_d_g_loss = []
global_epochs   = []

@tf.function
def train_step(images, target, latent_size, image_shape, discriminator_optimizer, generator_optimizer, learning_rate, add_noise: bool, beta_min, generator_model: tf.keras.Model, discriminator_model: tf.keras.Model, epoch):
    """
        Represents a training step (feed-forward + backpropagation & weights updating)
        Normalizes the images

        <Params>
            images                      = images from the data set
            target                      = labels from the data set
            latent_size                 = size of the latent vector
            image_shape                 = shape of the normalized images received from train()
            discriminator_optimizer     = optimizer to use for the discriminator training step (eg: 'Adam')
            generator_optimizer         = optimizer to use for the generator training step (eg: 'Adam')
            learning_rate               = learning rate
            add_noise                   = whether to add Gaussian noise to the real & generated images before passing to Discriminator
            beta_min                    = beta_1 value used for the optimizers
            generator_model             = configured & initialised generator model
            discriminator_model         = configured & initialised discriminator model
            epoch                       = current epoch for this training step (purely eval purposes)
    """
    # Update global epochs array
    global_epochs.append(epoch)

    # Get a random noise vector for the Generator
    noise = tf.random.normal([target.shape[0], latent_size])

    ############ TRAIN DISCRIMINATOR WITH REAL LABELS ############
    with tf.GradientTape() as discriminator_tape_real_labels:

        if add_noise == True:
            # Add noise to images used for DISCRIMINATOR & image generated by GENERATOR 
            # to introduce an element of stability to the data distributions [ref.6]
            noisy_real_images = images # use a copy of the original images tensor
            noisy_real_images = noisy_real_images + (0.1**0.5) * np.random.randn(image_shape[0], image_shape[1], 3)
            discriminator_real_output    = discriminator_model([noisy_real_images, target], training=True)
        else:
            discriminator_real_output    = discriminator_model([images, target], training=True)
            
        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)
        real_targets    = tf.ones_like(discriminator_real_output)
        loss            = discriminator_loss_function('binary_cross_entropy', real_targets, discriminator_real_output)
        global_d_loss.append(loss.numpy())
        print(f'LOSS (DISCRMINATOR [REAL LABELS]) : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_real_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate, beta_min)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

    ############ TRAIN DISCRIMINATOR WITH FAKE LABELS ############
    with tf.GradientTape() as discriminator_tape_fake_labels:

        generated_images             = generator_model([target, noise], training=True)

        if add_noise == True:
            # Add noise to generated images (see lines 45-46)
            noisy_generated_images = generated_images # use a copy of the original images tensor
            noisy_generated_images = noisy_generated_images + (0.1**0.5) * np.random.randn(image_shape[0], image_shape[1], 3)
            discriminator_fake_output    = discriminator_model([noisy_generated_images, target], training=True)
        else:
            discriminator_fake_output    = discriminator_model([generated_images, target], training=True)
        
        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)   
        fake_targets    = tf.zeros_like(discriminator_fake_output)
        loss            = discriminator_loss_function('binary_cross_entropy', fake_targets, discriminator_fake_output)
        global_d_g_loss.append(loss.numpy())
        print(f'LOSS (DISCRMINATOR [FAKE LABELS]) : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_fake_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate, beta_min)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

    ############ TRAIN GENERATOR WITH MATCHING (REAL) LABELS ############
    with tf.GradientTape() as generator_tape:

        generated_images    = generator_model([target, noise], training=True)

        if add_noise == True:
            # Add noise to generated images (see lines 45-46)
            noisy_generated_images = generated_images # use a copy of the original images tensor
            noisy_generated_images = noisy_generated_images + (0.1**0.5) * np.random.randn(image_shape[0], image_shape[1], 3)
            discriminator_fake_output    = discriminator_model([noisy_generated_images, target], training=True)
        else:
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
        plt.savefig('run/generated/trainingSample.png')
        plt.clf()

        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)   
        real_targets    = tf.ones_like(discriminator_fake_output)
        loss            = generator_loss_function('binary_cross_entropy', real_targets, discriminator_fake_output)
        print(f'LOSS (GENERATOR [REAL LABELS]) : {loss}')
    
        # Calculate gradient for discriminator training step with real labels
        gradients       = generator_tape.gradient(loss, generator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(generator_optimizer, learning_rate, beta_min)
        optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))
    
    # Return current losses & epoch history
    return global_d_loss, global_d_g_loss, global_epochs

def get_optimizer(optimizer, learning_rate, beta_min):
    """
        Gets a tf.keras.optimizers optimizer instance

        <Params>
            optimizer       = string argument for the optimizer to get an instance for (eg: 'Adam')
            learning_rate   = learning rate to configure the optimzier with
    """
    if optimizer == 'Adam':
        return optimizers.Adam(learning_rate=learning_rate, beta_1=beta_min, beta_2=0.9)
    elif optimizer == 'Adamax':
        return optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_min, beta_2=0.9, epsilon=1e-05)
        