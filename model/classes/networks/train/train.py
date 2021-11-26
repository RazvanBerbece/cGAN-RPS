#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from model.classes.networks.train.train_step import train_step
from model.functions.normalize import normalize
import time as time

# Evaluation imports
from model.classes.networks.train.evaluate import evaluate_model_loss

@tf.function
def train(dataset, shape, epochs, learning_rate, add_noise: bool, latent_size, beta_min, discriminator_optimizer, generator_optimizer, generator_model, discriminator_model):
    """
        Training wrapper function for train_step
        TODO: Could use zip() for passing the arguments (most train_step() arguments) to the function 

        <Params>
            dataset                     = dataset to train models on
            shape                       = shape of images used in training
            epochs                      = number of epochs for training
            learning_rate               = learning rate
            add_noise                   = whether to add Gaussian noise to the real & generated images before passing to Discriminator
            latent_size                 = size of the latent vector
            beta_min                    = beta_1 for optimizer used in updating network weights (applying gradients)
            discriminator_optimizer     = optimizer to use for the discriminator training step (eg: 'Adam')
            generator_optimizer         = optimizer to use for the generator training step (eg: 'Adam')
            generator_model             = configured & initialised generator model
            discriminator_model         = configured & initialised discriminator model
    """
    print("\n")
    for epoch in range(epochs):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"EPOCH {epoch + 1}/{epochs}\n")
        start = time.time()
        counter = 0
        for image_batch in dataset:
            # Index 0 of batch ~> Vector of images (128, 300, 300, 3)
            # Index 1 of batch ~> Vector of targets (128,) {ie: 0 = ROCK, 1 = PAPER, 2 = SCISSORS}
            print(f"\n~~~ PROCESSING BATCH {counter + 1}/{len(list(dataset))}... ~~~\n")
            counter += 1
            img_float32 = tf.cast(image_batch[0], dtype=tf.float32)
            normalized_imgs = normalize(img_float32, shape)
            d_loss, d_g_loss, epochs = train_step(normalized_imgs, image_batch[1], latent_size, shape, discriminator_optimizer, generator_optimizer, learning_rate, add_noise, beta_min, generator_model, discriminator_model, epoch+1)
        evaluate_model_loss(d_loss, d_g_loss, epochs)
        print('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))