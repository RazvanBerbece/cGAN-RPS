#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

class Generator:
    """
        Class that represents a cGAN Generator

        INPUTS
            1. Noise vector (call it z)
            2. Label (call it y)
            -----------------------------------
            G(z, y) = x|y (x conditioned on y, where x = generated fake example)

        OUTPUT
            Generates a class-restricted sample
    """

    def __init__(self):
        """
            <<Constructor>>
        """
        self.model       = None
        self.input_label = layers.Input(shape=(1,)) # input layer for the label 
        """
            Experiments suggest that the distribution of the noise doesn't matter much, 
            so we can choose something that's easy to sample from, like a uniform distribution. 
            For convenience the space from which the noise is sampled is usually 
            of smaller dimension than the dimensionality of the output space.
        """
        self.input_noise = layers.Input(shape=(100,)) # input layer for the noise (as in any other GAN)

    def activate_label_input(self, num_classes, embedding_size, num_nodes):
        """
            Gets initial output from the label input nodes

            <Params>
                num_classes      = number of classes (eg: 3 [rock, paper, scissors])
                embedding_size   = size of embedding (see NLP common practices)
                num_nodes        = number of nodes for the fully connected Dense layer (ie: '3x3', '4x4', '4x5')
        """
        # Get node structure from argument
        # TODO: Find out why there is a dependency on the number of nodes for the reshaped layer
        #       1. For merging of the noise matrix output after activations & label output after activations
        sizes = num_nodes.split("x")
        self.embedding_row_size = int(sizes[0])
        self.embedding_col_size = int(sizes[1])
        # Network Layers & Forwarding
        output_label_embedded   = layers.Embedding(num_classes, embedding_size)(self.input_label) # embed the categorical input
        output_label_dense      = layers.Dense(self.embedding_row_size * self.embedding_col_size)(output_label_embedded) # basic matrix multiplication of form output = activation(dot(input, kernel) + bias)
        output_label_reshape    = layers.Reshape((self.embedding_row_size, self.embedding_col_size, 1))(output_label_dense)
        return output_label_reshape

    def activate_noise_input(self, num_nodes):
        """
            Gets initial output from the noise vector input nodes

            <Params>
                num_nodes  = number of nodes for the fully connected Dense layer
        """
        # Network Layers & Forwarding
        output_noise_dense      = layers.Dense(num_nodes * self.embedding_row_size * self.embedding_col_size)(self.input_noise)
        output_noise_relu       = layers.ReLU()(output_noise_dense)
        output_noise_reshape    = layers.Reshape((self.embedding_row_size, self.embedding_col_size, num_nodes))(output_noise_relu)
        return output_noise_reshape

    def process_generator_network(self, num_classes, embedding_size, label_num_nodes, noise_num_nodes, generator_initial_num_nodes):
        """
            Sets the Generator's ML model
            Defines the layer architecture using the merged outputs of the label & noise matrixes inputs
            Currently supports 4 hidden layers
            TODO: Dynamic number of layers

            <Params>
                num_classes                 = number of classes (eg: 3 [rock, paper, scissors])
                embedding_size              = size of embedding (see NLP common practices)
                label_num_nodes             = number of nodes for the fully connected Dense layer (ie: '3x3', '4x4', '4x5')
                noise_num_nodes             = number of nodes for the fully connected Dense layer
                generator_initial_num_nodes = initial number of nodes for the CNN layers (from argument value decrements as num x K, num x (K / 2), ..., num x 1)
        """
        # Merge label + noise output matrices
        initial_label_output    = self.activate_label_input(num_classes, embedding_size, label_num_nodes)
        initial_noise_output    = self.activate_noise_input(noise_num_nodes) # HAS to run AFTER activate_label_input()
        merged_net_input        = layers.Concatenate()([initial_noise_output, initial_label_output])
        # Generator CNN Architecture + Forwarding
        # x is the input that is fordward fed through the network
        x = layers.Conv2DTranspose(           \
             generator_initial_num_nodes * 8, \
             kernel_size=4,                   \
             strides=2,                       \
             padding='same',                  \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), \
             use_bias=False,                  \
             name='conv_transpose_1')(merged_net_input)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
        x = layers.ReLU(name='relu_1')(x)

        x = layers.Conv2DTranspose(           \
             generator_initial_num_nodes * 4, \
             kernel_size=4,                   \
             strides=2,                       \
             padding='same',                  \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), \
             use_bias=False,                  \
             name='conv_transpose_2')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
        x = layers.ReLU(name='relu_2')(x)

        x = layers.Conv2DTranspose(           \
             generator_initial_num_nodes * 2, \
             kernel_size=4,                   \
             strides=2,                       \
             padding='same',                  \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), \
             use_bias=False,                  \
             name='conv_transpose_3')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
        x = layers.ReLU(name='relu_3')(x)

        x = layers.Conv2DTranspose(           \
             generator_initial_num_nodes,     \
             kernel_size=4,                   \
             strides=2,                       \
             padding='same',                  \
             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), \
             use_bias=False,                  \
             name='conv_transpose_4')(x)
        x = layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02, name='bn_4')(x)
        x = layers.ReLU(name='relu_4')(x)

        # Output
        output = layers.Conv2DTranspose(      \
            3,                                \
            kernel_size=4,                    \
            strides=2,                        \
            padding='same',                   \
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), \
            use_bias=False,                   \
            activation='tanh',                \
            name='conv_transpose_5')(x)

        # Set Generator Model
        self.model = tf.keras.Model([self.input_label, self.input_noise], output)
