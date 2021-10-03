#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from model.classes.networks.train.train_step import train_step

"""
    Training wrapper function for train_step 

    <Params>
        dataset = dataset to train models on (shuffled and )
        epochs  = number of epochs for training
"""
def train(dataset, epochs):
    pass