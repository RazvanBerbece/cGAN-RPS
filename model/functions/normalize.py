#!/usr/bin/env python3

# Package Imports
import tensorflow as tf

"""
    Normalizes tensors used for the purpose of cGAN-RPS with the given shape
    Also changes all colour values from [0, 255.0] to [-1, 1]
"""
@tf.function
def normalize(tensor, shape, ):
    tensor = tf.image.resize(tensor, shape)
    tensor = tf.subtract(tf.divide(tensor, 172.5), 1) # 255.0 / 2 = 172.5
    return tensor