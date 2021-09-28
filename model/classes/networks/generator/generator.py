#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from tensorflow.keras import layers

class Generator:
    """
        Class that represents a cGAN Generator
    """

    """
        <<Constructor>>
    """
    def __init__(self):
        self.input_label = layers.Input(shape=(1,)) # input layer for the label 
        """
            Experiments suggest that the distribution of the noise doesn't matter much, 
            so we can choose something that's easy to sample from, like a uniform distribution. 
            For convenience the space from which the noise is sampled is usually 
            of smaller dimension than the dimensionality of the output space.
        """
        self.input_noise = layers.Input(shape=(100,)) # input layer for the noise (as in any other GAN)

    """
        Gets output for the label input nodes

        <Params>
        num_classes      = number of classes (eg: 3 [rock, paper, scissors])
        embedding_size   = size of embedding (see NLP common practices)
        num_nodes        = number of nodes for the fully connected Dense layer
    """
    def activate_label_input(self, num_classes, embedding_size, num_nodes):
        output_label_embedded   = layers.Embedding(num_classes, embedding_size)(self.input_label) # embed the categorical input
        output_label_dense      = layers.Dense(num_nodes)(output_label_embedded) # basic matrix multiplication of form output = activation(dot(input, kernel) + bias)
        output_label_reshape    = layers.Reshape((4, 4, 1))(output_label_dense) 
        return output_label_reshape

    """
        Gets output for the noise input nodes

        <Params>
        num_nodes  = number of nodes for the fully connected Dense layer 
    """
    def activate_noise_input(self, num_nodes):
        output_noise_dense      = layers.Dense(num_nodes)(self.input_noise)
        output_noise_relu       = layers.ReLU()(output_noise_dense)
        output_noise_reshape    = layers.Reshape()(output_noise_relu)
        return output_noise_reshape