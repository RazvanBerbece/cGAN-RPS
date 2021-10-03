#!/usr/bin/env python3

# Package Imports
import os
# from model.classes.prep.Dataset import Dataset
from model.classes.networks.generator.Generator import Generator
from model.classes.networks.discriminator.Discriminator import Discriminator
from model.classes.networks.train.train import train

# Data Prepping
# dataset = Dataset(dataset='RockPaperScissors', seed=512, trainRatio='75%', batchSize=128)

# CONSTANTS
num_classes                     = 3
latent_size                     = 100

# HYPERPARAMETERS
# Generator Params
generator_embedding_size        = 100
label_num_nodes                 = '4x4'
noise_num_nodes                 = 512
generator_initial_num_nodes     = 64
# Discriminator Params
discriminator_image_shape       = (128, 128, 3)
discriminator_embedding_size    = 100
discriminator_initial_num_nodes = 64
dropout_rate                    = 0.4
activation                      = 'sigmoid'
# Training Step Params
discriminator_optimiser         = 'Adam'
generator_optimiser             = 'Adam'
# Training Params
epochs                          = 25
learning_rate                   = 0.05

# Generator Init & Config
generator = Generator()
generator.process_generator_network(                                    \
    num_classes=num_classes,                                            \
    embedding_size=generator_embedding_size,                            \
    label_num_nodes=label_num_nodes,                                    \
    noise_num_nodes=noise_num_nodes,                                    \
    generator_initial_num_nodes=generator_initial_num_nodes)

# Discriminator Init & Config
discriminator = Discriminator(input_image_shape=discriminator_image_shape)
discriminator.process_discriminator_network(                            \
    num_classes=num_classes,                                            \
    embedding_size=discriminator_embedding_size,                        \
    discriminator_initial_num_nodes=discriminator_initial_num_nodes,    \
    dropout_rate=dropout_rate,                                          \
    activation=activation)

# Training


# Generate Image Using Trained cGAN model
# TODO