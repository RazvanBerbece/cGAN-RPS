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

        <Params>
        num_classes     = number of classes (labels; eg: 3 [rock, paper, scissors])
        num_embedding   = embedding size (TODO: could be fine-tuned ? To find out more)
    """
    def __init__(self, num_classes, num_embedding):
        self.num_classes = num_classes
        self.label_input = layers.Input(shape=(1,))
        self.noise_input = layers.Input(shape=(num_embedding,))

    def activate_label(self):
