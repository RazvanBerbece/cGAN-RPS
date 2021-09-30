#!/usr/bin/env python3

# Package Imports
import os
# from model.classes.prep.Dataset import Dataset
from model.classes.networks.generator.Generator import Generator

# Data Prepping
# dataset = Dataset(dataset='RockPaperScissors', seed=512, trainRatio='75%', batchSize=128)

# CONSTANTS
num_classes = 3

# HYPERPARAMETERS
# Generator
embedding_size              = 100
label_num_nodes             = '4x4'
noise_num_nodes             = 512
generator_initial_num_nodes = 64
# Discriminator
# TODO
# Training
# TODO

# Generator Init & Config
generator = Generator()
generator_model = generator.get_generator_network(           \
    num_classes=num_classes,                                 \
    embedding_size=embedding_size,                           \
    label_num_nodes=label_num_nodes,                         \
    noise_num_nodes=noise_num_nodes,                         \
    generator_initial_num_nodes=generator_initial_num_nodes)
print(generator_model)

# Discriminator Init & Config
# TODO

# Training
# TODO

# Generate Image Using Trained cGAN model
# TODO