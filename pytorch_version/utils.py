import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import torch
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load the dataset into a NumPy array.

    Args:
        path: str, path of the dataset

    Returns:
        float: accuracy of the model
    """
    try:
        data = np.genfromtxt(path, delimiter=",", dtype=str)
        return data

    except FileNotFoundError:
        return None
    except IsADirectoryError:
        return None


def save_network(network, path="models/pytorch_model.pth"):
    os.makedirs("models", exist_ok=True)
    torch.save(network.state_dict(), path)
    print(f"Model successfully saved to {path}.")


def draw_loss_accu(network):
    os.makedirs("visuals", exist_ok=True)
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(network.train_losses, label="Training loss")
    ax1.plot(network.val_losses, label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation loss")
    ax1.legend()
    ax1.text(0.05, 0.05, network.params, transform=ax1.transAxes, fontsize=10, verticalalignment='bottom')

    ax2.plot(network.train_accu, label="Training accuracy")
    ax2.plot(network.val_accu, label="Validation accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation accuracy")
    ax2.legend()

    plt.savefig(f"visuals/model_{network.id}_loss_accuracy.png")
    plt.close(fig)


def draw_comp_accuracy(networks):
    os.makedirs("visuals", exist_ok=True)
    plt.clf()
    plt.figure(figsize=(10, 6))

    for network in networks:
        x_value = [(epoch / (len(network.val_accu) - 1)) * 100 for epoch in range(len(network.val_accu))]
        plt.plot(x_value, network.val_accu, label=f"#{network.id} Validation accuracy")

    plt.xlabel("Training Progress (%)")
    plt.ylabel("Accuracy")
    plt.title("Models' accuracy comparison")
    plt.legend()
    plt.savefig("visuals/accuracy_comparison.png")


def draw_comp_loss(networks):
    os.makedirs("visuals", exist_ok=True)
    plt.clf()
    plt.figure(figsize=(10, 6))

    for network in networks:
        x_value = [(epoch / (len(network.val_losses) - 1)) * 100 for epoch in range(len(network.val_losses))]
        plt.plot(x_value, network.val_losses, label=f"#{network.id} Validation loss")

    plt.xlabel("Training Progress (%)")
    plt.ylabel("Loss")
    plt.title("Models' loss comparison")
    plt.legend()
    plt.savefig("visuals/loss_comparison.png")
