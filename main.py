#!/usr/bin/env python3

# Package Imports
import sys
from model.classes.prep.Dataset import Dataset
from model.classes.networks.generator.Generator import Generator
from model.classes.networks.discriminator.Discriminator import Discriminator
from model.functions.train.train import train
from tensorflow import keras

# Data Preparation
seed                            = 345
batch_size                      = 256
train_ratio                     = '70%'
dataset = Dataset(dataset='RockPaperScissors', seed=seed, trainRatio=train_ratio, batchSize=batch_size)

# MODELLING CONSTANTS
num_classes                     = 3
latent_size                     = 100
training_image_shape            = (128, 128)

# HYPERPARAMETERS
# Generator Params
generator_embedding_size        = 5
label_num_nodes                 = '4x4'
noise_num_nodes                 = 512
generator_initial_num_nodes     = 128
# Discriminator Params
discriminator_image_shape       = (128, 128, 3)
discriminator_embedding_size    = 5
discriminator_initial_num_nodes = 128
dropout_rate                    = 0.2
# Training Step Params
discriminator_optimiser         = 'Adamax'
generator_optimiser             = 'Adamax'
# Training Params
epochs                          = 100         # (TODO: OPTIMISE PROCESS ?? (HYPERPARAMS, train_step()))
learning_rate_discriminator     = 0.0002      # for this dataset & problem space, learning rates close to 0 prevent GAN COLLAPSE [ref.6]
learning_rate_generator         = 0.0002      # for this dataset & problem space, learning rates close to 0 prevent GAN COLLAPSE [ref.6]
beta_min                        = 0.5
add_noise                       = True        # Adds Gaussian noise to image batch when training

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
    dropout_rate=dropout_rate)                                              

# Training
arguments = sys.argv
if arguments[1] == '-fromScratch':  
    """
        Train a new cGAN model with new, random weights
    """
    print('Training new cGAN model from scratch...')
    train(                                                                      \
        dataset=dataset.dataset,                                                \
        shape=training_image_shape,                                             \
        epochs=epochs,                                                          \
        learning_rate_d=learning_rate_discriminator,                            \
        learning_rate_g=learning_rate_generator,                                \
        add_noise=add_noise,                                                    \
        latent_size=latent_size,                                                \
        beta_min=beta_min,                                                      \
        discriminator_optimizer=discriminator_optimiser,                        \
        generator_optimizer=generator_optimiser,                                \
        generator_model=generator.model,                                        \
        discriminator_model=discriminator.model                                 \
    )
elif arguments[1] == '-resume':
    """
        Resume training of an existing cGAN model from trained_models/
    """
    print('Loading existing cGAN model...')
    # Load models from trained_models/
    model_g = keras.models.load_model('trained_models/generator', compile=False)
    model_d = keras.models.load_model('trained_models/discriminator', compile=False)

    # Train loaded models
    train(                                                                      \
        dataset=dataset.dataset,                                                \
        shape=training_image_shape,                                             \
        epochs=epochs,                                                          \
        learning_rate_d=learning_rate_discriminator,                            \
        learning_rate_g=learning_rate_generator,                                \
        add_noise=add_noise,                                                    \
        latent_size=latent_size,                                                \
        beta_min=beta_min,                                                      \
        discriminator_optimizer=discriminator_optimiser,                        \
        generator_optimizer=generator_optimiser,                                \
        generator_model=model_g,                                                \
        discriminator_model=model_d                                             \
    )
    