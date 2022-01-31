#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
import numpy as np

class ImageGenerator:
    """
        Class that handles image generation using the trained cGAN model
    """

    def __init__(self, model):
        """
            Initialise self.model with the trained Keras model
        """
        self.model = model

    def get_model_summary(self):
        """
            Display to stdout the summary of the Keras model attached to self.model
            Returns a pair (status, errMsg), where status is whether the operation has been successful
        """
        try:
            self.model.summary()
            return (True, '')
        except ValueError:
            return (False, 'ValueError : Summary method cannot be used. Function called before model is built.')
    
    def generate_image(self, target: str):
        """
            Generate 1 image using the Generator model G in self.model with a given target
            Returns an np array representation of the generated image

            <Params>
                target = a str with the target of the generator (sanitised before being passed, eg: 'rock')
        """

        # Get numerical index for str target argument
        numerical_target = -1
        if target == 'rock':
            numerical_target = 0
        elif target == 'paper':
            numerical_target = 1
        else:
            numerical_target = 2
        np_target = np.array([[numerical_target]]) # shape (1,)

        # Get a random noise vector to pass to the Generator
        latent_size = 100 # TODO: Figure out how to remove dependency of latent_size on generator API run
        noise = tf.random.normal([np_target.shape[0], latent_size])

        # Run a feedforward of the noise + target concat through G
        generated_images = self.model([np_target, noise], training=False)

        # Bring to normal format (colour, size, etc)
        generated_image_item = generated_images[0]
        generated_image_item = 255 * generated_image_item
        generated_image_item = tf.cast(generated_image_item, tf.uint8)

        # Cast EagerTensor to np array for ease of further casting
        generated_image_item = np.array(generated_image_item)

        return generated_image_item
