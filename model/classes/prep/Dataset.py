#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    """ Class used for the dataset reading & configuration phase of data preparation. """

    """
        <<Constructor>>

        <Params>
        dataset     = TFDS dataset name
        seed        = Seed for random shuffling of the data
        trainRatio  = Training data ratio (remaining is test data)
        batchSize   = Size of each batch of data
    """
    def __init__(self, dataset, seed, trainRatio, batchSize):

        # as_supervised=True returns a dataset that has a 2-tuple structure (input, label)
        self.dataset = tfds.load(dataset, split=f'train[:{trainRatio}]', as_supervised=True, shuffle_files=True)

        # Shuffle the data with a seed and create batches of a specific size
        self.dataset = self.dataset.shuffle(seed).batch(batchSize)