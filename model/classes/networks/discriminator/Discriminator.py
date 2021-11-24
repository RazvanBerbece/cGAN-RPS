#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

class Discriminator:
    """
        Class that represents a cGAN Discriminator
        Fed Real (training set) & Fake (generator outputs) examples with labels

        RECOGNIZES
            1. Real & Fake examples
            2. Matching pairs -- examples where the image matches the label
        
        OUTPUT
            Probability that the example is real or fake
    """

    def __init__(self, input_image_shape):
        """
            <<Constructor>>

            <Params>
                input_image_shape = shape of the discriminator image inputs
        """
        self.model       = None
        self.input_label = layers.Input(shape=(1,)) # input layer for the label 
        self.input_image = layers.Input(shape=input_image_shape) # input layer for the noise (as in any other GAN)

    @tf.function
    def activate_label_input(self, num_classes, embedding_size):
        """
            Gets initial output from the label input nodes (Discriminator)

            <Params>
                num_classes      = number of classes (eg: 3 [rock, paper, scissors])
                embedding_size   = size of embedding (see NLP common practices)
        """
        num_nodes = self.input_image.shape[1] * self.input_image.shape[2] * self.input_image.shape[3] # used in a dense layer to scale up to input image shape
        # Network Layers & Forwarding
        output_label_embedded   = layers.Embedding(num_classes, embedding_size)(self.input_label) # embedding for label input
        output_label_dense      = layers.Dense(num_nodes)(output_label_embedded)
        output_label_reshape    = layers.Reshape((self.input_image.shape[1], self.input_image.shape[2], 3))(output_label_dense) # get output in tensor shape
        return output_label_reshape
    
    def activate_image_input(self):
        """
            Gets initial output from the image input nodes (Discriminator)
        """
        return self.input_image
        
    @tf.function
    def process_discriminator_network(self, num_classes, embedding_size, discriminator_initial_num_nodes, dropout_rate, activation):
        """
            Sets the Discriminator's ML model
            Defines the layer architecture using the merged outputs of the label & image vector inputs
            Currently supports 4 hidden layers
            TODO: Dynamic number of layers

            <Params>
                num_classes                     = number of classes (eg: 3 [rock, paper, scissors])
                embedding_size                  = size of embedding (see NLP common practices)
                discriminator_initial_num_nodes = initial number of nodes for the CNN layers (from argument value increments as num, num x 2, ..., num x K)
                dropout_rate                    = dropout rate for the dropout layers of the Discriminator CNN (see Dropout doc)
                activation                      = activation function for Dense output layer (1 node) (eg: 'sigmoid', as we want a probability)
        """
        # Merge label + noise output matrices
        initial_label_output    = self.activate_label_input(num_classes, embedding_size)
        initial_image_output    = self.activate_image_input() # HAS to run AFTER activate_label_input()
        merged_net_input        = layers.Concatenate()([initial_image_output, initial_label_output])
        # Discriminator CNN Architecture + Forwarding
        # x is the input that is fordward fed through the network
        x = layers.Conv2D(                                                                  \
             discriminator_initial_num_nodes,                                               \
             kernel_size=4,                                                                 \
             strides=2,                                                                     \
             padding='same',                                                                \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),  \
             use_bias=False,                                                                \
             name='conv_transpose_1')(merged_net_input)
        x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)

        x = layers.Conv2D(                                                                  \
             discriminator_initial_num_nodes * 2,                                           \
             kernel_size=4,                                                                 \
             strides=2,                                                                     \
             padding='same',                                                                \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),  \
             use_bias=False,                                                                \
             name='conv_transpose_2')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
        x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)

        x = layers.Conv2D(                                                                  \
             discriminator_initial_num_nodes * 4,                                           \
             kernel_size=4,                                                                 \
             strides=2,                                                                     \
             padding='same',                                                                \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),  \
             use_bias=False,                                                                \
             name='conv_transpose_3')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8,  center=1.0, scale=0.02, name='bn_2')(x)
        x = layers.LeakyReLU(0.2, name='leaky_relu_3')(x)

        x = layers.Conv2D(                                                                  \
             discriminator_initial_num_nodes * 8,                                           \
             kernel_size=4,                                                                 \
             strides=2,                                                                     \
             padding='same',                                                                \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),  \
             use_bias=False,                                                                \
             name='conv_transpose_4')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
        x = layers.LeakyReLU(0.2, name='leaky_relu_4')(x)

        # Flatten
        flattened_x         = layers.Flatten()(x)
        # Dropout
        dropout_flattened   = layers.Dropout(dropout_rate)(flattened_x)
        # Output
        dense_dropout       = layers.Dense(1, activation=activation)(dropout_flattened)

        # Set Discriminator Model
        self.model = tf.keras.Model([initial_image_output, self.input_label], dense_dropout)