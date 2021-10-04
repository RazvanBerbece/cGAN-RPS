#!/usr/bin/env python3

# Package Imports
import os
from model.classes.prep.Dataset import Dataset
from model.classes.networks.generator.Generator import Generator
from model.classes.networks.discriminator.Discriminator import Discriminator
from model.classes.networks.train.train import train

# Data Preparation
seed                            = 512
batch_size                      = 128
train_ratio                     = '75%'
dataset = Dataset(dataset='RockPaperScissors', seed=seed, trainRatio=train_ratio, batchSize=batch_size)

# MODELLING CONSTANTS
num_classes                     = 3
latent_size                     = 100
training_image_shape            = (128, 128)

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
activation                      = 'sigmoid' # 
# Training Step Params
discriminator_optimiser         = 'Adam'
generator_optimiser             = 'Adam'
# Training Params
epochs                          = 25 # ~211 sec per epoch (TODO: OPTIMISE PROCESS ?? (HYPERPARAMS, train_step()))
learning_rate                   = 0.5

# Generator Init & Config
generator = Generator()
generator.process_generator_network(                                        \
    num_classes=num_classes,                                                \
    embedding_size=generator_embedding_size,                                \
    label_num_nodes=label_num_nodes,                                        \
    noise_num_nodes=noise_num_nodes,                                        \
    generator_initial_num_nodes=generator_initial_num_nodes)

# Discriminator Init & Config
discriminator = Discriminator(input_image_shape=discriminator_image_shape)
discriminator.process_discriminator_network(                                \
    num_classes=num_classes,                                                \
    embedding_size=discriminator_embedding_size,                            \
    discriminator_initial_num_nodes=discriminator_initial_num_nodes,        \
    dropout_rate=dropout_rate,                                              \
    activation=activation)

# Training
train(                                                                      \
    dataset=dataset.dataset,                                                \
    shape=training_image_shape,                                             \
    epochs=epochs,                                                          \
    learning_rate=learning_rate,                                            \
    latent_size=latent_size,                                                \
    discriminator_optimizer=discriminator_optimiser,                        \
    generator_optimizer=generator_optimiser,                                \
    generator_model=generator.model,                                        \
    discriminator_model=discriminator.model                                 \
)

# Generate Image Using Trained cGAN model
# TODO