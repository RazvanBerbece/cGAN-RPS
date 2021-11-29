#!/usr/bin/env python3

# Package Imports
from matplotlib import pyplot as plt

def evaluate_model_loss(d_loss, d_g_loss, epochs):
    """
        Plot the losses of the discriminator model (discriminator training on real data & discriminator training on generated data)
    """

    # Stdout losses & epoch list
    print('\n', d_loss)
    print(d_g_loss)
    print(epochs, '\n')

    # Plot interface settings
    plt.figure(figsize=(15, 6), dpi=80)
    plt.xticks(range(1, epochs[-1] + 1))

    # Plot data visualisation config
    plt.plot(epochs, d_loss)
    plt.plot(epochs, d_g_loss)

    # Plot UX 
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Discriminator (On dataset tensors)", "Discriminator (On generated tensors)"])

    # Store plot to local & clear plt figure
    plt.savefig("temp/train_history/evaluation.png")
    plt.clf()