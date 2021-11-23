#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from model.classes.networks.generator.Generator import Generator
from model.classes.networks.discriminator.Discriminator import Discriminator

class ImageGenerator:
    """
        Class that handles image generation using the trained cGAN model
    """

    def __init__(self, model: Generator):
        self.model = model
    
    def generate_image(self, noise, num_classes):
        pass
    
