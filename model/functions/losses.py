#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

def discriminator_loss_function(function, label, output):
    """
        Loss function for the discriminator which is used in the training phase
        The loss is calculated with real targets (eg: 'paper')
        TODO: Implement dynamic loss function support

        <Params>
            function    = loss function used for the generator instance
            label       = 
            output      = 
    """
    if function == 'binary_cross_entropy':
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
        discriminator_loss = binary_cross_entropy(label, output)
        return discriminator_loss

def generator_loss_function(function, label, fake_output):
    """
        Loss function for the generator which is used in the training phase
        The loss is calculated with real targets (eg: 'paper')
        TODO: Implement dynamic loss function support

        <Params>
            function    = loss function used for the generator instance
            label       = 
            fake_output = 
    """
    if function == 'binary_cross_entropy':
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
        generator_loss = binary_cross_entropy(label, fake_output)
        return generator_loss