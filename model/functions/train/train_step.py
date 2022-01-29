#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_math_ops import add
from model.functions.loss.losses import discriminator_loss_function, generator_loss_function
from matplotlib import pyplot as plt
import numpy as np

"""
    Fix for Issue #5 (Git)
    Source : https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio
"""
tf.config.run_functions_eagerly(True)

@tf.function
def train_step(images, target, latent_size, image_shape, discriminator_optimizer, generator_optimizer, learning_rate_d, learning_rate_g, add_noise: bool, beta_min, generator_model: tf.keras.Model, discriminator_model: tf.keras.Model):
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
            learning_rate_d             = learning rate for discriminator gradient
            learning_rate_g             = learning rate for generator gradient
            add_noise                   = whether to add Gaussian noise to the real & generated images before passing to Discriminator
            beta_min                    = beta_1 value used for the optimizers
            generator_model             = configured & initialised generator model
            discriminator_model         = configured & initialised discriminator model
    """
    # Loss variables to use for evaluation
    d_loss      = None
    d_g_loss    = None
    g_loss      = None

    # Get a random noise vector for the Generator
    noise       = tf.random.normal([target.shape[0], latent_size])

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
        d_loss = loss.numpy()
        print(f'd_loss : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_real_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate_d, beta_min)
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
        d_g_loss = loss.numpy()
        print(f'd_g_loss : {loss}')

        # Calculate gradient for discriminator training step with real labels
        gradients       = discriminator_tape_fake_labels.gradient(loss, discriminator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(discriminator_optimizer, learning_rate_d, beta_min)
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

        # Extract labels from first 9 targets of input item
        generated_image_labels = []
        for label in target:
            display_label = ""
            if (label == 0):
                display_label = "ROCK"
            elif (label == 1):
                display_label = "PAPER"
            else:
                display_label = "SCISSORS"
            generated_image_labels.append(display_label)

        # Get the first 9 generated images and cast them
        generated_image_items = generated_images[:9]
        generated_image_items = 255 * generated_image_items
        generated_image_items = tf.cast(generated_image_items, tf.uint8)

        # Plot the 9 images
        counter = 0
        plt.figure()
        plt.title(display_label)
        _, axarr = plt.subplots(3,3)
        for row in axarr:
            for col in row:
                col.imshow(generated_image_items[counter])
                col.set_title(generated_image_labels[counter])
                counter = counter + 1
        plt.savefig("temp/generated/trainingSampleMulti.png")
        # Clear plt figure (so eval can use it)
        plt.clf()

        # 1-Tensor of the same shape as the discriminator outputs
        # (all training images are real, so that's why we use a 1-Tensor)   
        real_targets    = tf.ones_like(discriminator_fake_output)
        loss            = generator_loss_function('binary_cross_entropy', real_targets, discriminator_fake_output)
        g_loss          = loss.numpy()
        print(f'g_loss : {loss}')
    
        # Calculate gradient for discriminator training step with real labels
        gradients       = generator_tape.gradient(loss, generator_model.trainable_variables)

        # Optimise parameters
        optimizer       = get_optimizer(generator_optimizer, learning_rate_g, beta_min)
        optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))
    
    # Return current losses & epoch history
    return d_loss, d_g_loss, g_loss

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
        