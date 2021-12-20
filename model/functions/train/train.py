#!/usr/bin/env python3

# Package Imports
import tensorflow as tf
from model.functions.train.train_step import train_step
from model.functions.transforms.normalize import normalize
import time as time

# Evaluation imports
from model.functions.eval.evaluate import evaluate_model_loss

@tf.function
def train(dataset, shape, epochs, learning_rate_d, learning_rate_g, add_noise: bool, latent_size, beta_min, discriminator_optimizer, generator_optimizer, generator_model, discriminator_model):
    """
        Training wrapper function for train_step
        TODO: Could use zip() for passing the arguments (most train_step() arguments) to the function 

        <Params>
            dataset                     = dataset to train models on
            shape                       = shape of images used in training
            epochs                      = number of epochs for training
            learning_rate_d             = learning rate for discriminator gradient
            learning_rate_g             = learning rate for generator gradient
            add_noise                   = whether to add Gaussian noise to the real & generated images before passing to Discriminator
            latent_size                 = size of the latent vector
            beta_min                    = beta_1 for optimizer used in updating network weights (applying gradients)
            discriminator_optimizer     = optimizer to use for the discriminator training step (eg: 'Adam')
            generator_optimizer         = optimizer to use for the generator training step (eg: 'Adam')
            generator_model             = configured & initialised generator model
            discriminator_model         = configured & initialised discriminator model
    """
    print("\n")

    # Loss history for evaluation
    epochs_for_eval     = []
    losses_d_for_eval   = []
    losses_d_g_for_eval = []
    losses_g_for_eval   = []
    
    start_total = time.time()
    for epoch in range(epochs):

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"EPOCH {epoch + 1}/{epochs}\n")

        start = time.time()
        epochs_for_eval.append(epoch + 1)
        counter = 0

        # Eval variables
        epoch_wide_loss_d         = 0
        epoch_wide_loss_d_g       = 0
        epoch_wide_loss_g         = 0
        epoch_wide_losses_counter = 0

        # Train on each batch of given size in the dataset
        for image_batch in dataset:

            print(f"\n~~~ PROCESSING BATCH {counter + 1}/{len(list(dataset))}... ~~~\n")
            counter += 1

            # Index 0 of batch ~> Vector of images (128, 300, 300, 3)
            # Index 1 of batch ~> Vector of targets (128,) {ie: 0 = ROCK, 1 = PAPER, 2 = SCISSORS}
            img_float32 = tf.cast(image_batch[0], dtype=tf.float32)
            normalized_imgs = normalize(img_float32, shape)

            d_loss, d_g_loss, g_loss = train_step(normalized_imgs, image_batch[1], latent_size, shape, discriminator_optimizer, generator_optimizer, learning_rate_d, learning_rate_g, add_noise, beta_min, generator_model, discriminator_model)

            # Calculate total error per epoch for all batches
            epoch_wide_loss_d = epoch_wide_loss_d + d_loss
            epoch_wide_loss_d_g = epoch_wide_loss_d_g + d_g_loss
            epoch_wide_loss_g = epoch_wide_loss_g + g_loss
            epoch_wide_losses_counter = epoch_wide_losses_counter + 1
        
        # Append average loss of batches to the globals
        losses_d_for_eval.append(epoch_wide_loss_d / epoch_wide_losses_counter)
        losses_d_g_for_eval.append(epoch_wide_loss_d_g / epoch_wide_losses_counter)
        losses_g_for_eval.append(epoch_wide_loss_g / epoch_wide_losses_counter)

        # Plot losses
        evaluate_model_loss(losses_d_for_eval, losses_d_g_for_eval, losses_g_for_eval, epochs_for_eval)

        print('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    print('Total time for training {} epochs is {} sec\n'.format(epochs, time.time()-start_total))

    # Store models
    generator_model.save('trained_models/generator')
    discriminator_model.save('trained_models/discriminator')
