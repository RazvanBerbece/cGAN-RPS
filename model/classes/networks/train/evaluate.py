#!/usr/bin/env python3

# Package Imports
from matplotlib import pyplot as plt

def evaluate_model_loss(d_loss, d_g_loss, epochs):
    print(d_loss)
    print(epochs)
    plt.plot(d_loss, epochs)
    # plt.plot(d_g_loss, [25], color="red")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.legend(["Discriminator", "Discriminator (On generated tensors)"], loc="center right")
    plt.savefig('run/train_history/evaluation.png')