import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
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


def normalize_data(data):
    """
    Normalize the dataset

    Args:
        data: np.array, dataset

    Returns:
        np.array: normalized dataset
    """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    norm_data = (data - mean) / std_dev
    return norm_data, mean, std_dev


def normalize_data_spec(data, mean, std_dev):
    """
    Normalize the dataset with a given mean and std_dev.

    Args:
        data: np.array, dataset
        mean: float, mean used to normalize the training set of the model
        std_dev: float, standard deviation used to normalize the training set of the model

    Returns:
        np.array: normalized dataset
    """
    norm_data = (data - mean) / std_dev
    return norm_data


def one_hot(y_true, n_outputs):
    """
    Returns one hot version of y_true.

    Args:
        y_true (np.array): "true" value, what the model is supposed to predict.
        n_outputs: int, number of outputs of the current layer

    Returns:
        np.array: one hot version of y_true
    """
    one_hot = np.zeros((len(y_true), n_outputs))
    one_hot[np.arange(len(y_true)), y_true] = 1
    return one_hot


def save_network(network):
    """
    Saves the network in models/
    
    Args:
        network: Object of class Network defined in models.py
    """
    os.makedirs("models", exist_ok=True)
    files = os.listdir("models")
    model_files = [f for f in files if re.match(r"model#\d+\.pkl", f)]
    network.id = len(model_files) + 1

    with open(f"models/model#{network.id}.pkl", "wb") as f:
        pickle.dump(network, f)
    print(f"Network successfully saved in models/model#{network.id}.pkl.")


def load_network():
    """
    Loads the network of the given model number

    note: Yes I should only give the path of the network to this function
    and not take and validate input.
    """

    nb_model = None
    while (nb_model is None):
        nb_model = input("Model number: ")
        if (nb_model == "q"):
            return None
        try:
            print(f"Loading model#{nb_model}...")
            with open(f"models/model#{nb_model}.pkl", "rb") as f:
                network = pickle.load(f)
            print(f"model#{nb_model} successfully loaded.\n")
        except FileNotFoundError:
            print(f"Error: Couldn't load model#{nb_model}.\n")
            nb_model = None

    return network


def load_networks():
    """
    Loads every network from the models/ directory
    """
    networks = []

    files = os.listdir("models")
    model_files = [f for f in files if re.match(r"model#\d+\.pkl", f)]
    for model in model_files:
        with open(f"models/{model}", "rb") as f:
            networks.append(pickle.load(f))
            print(f"{model} successfully loaded.")

    return networks


def binary_cross_entropy(predictions, y_true):
    """
    Calculates the Binary Cross Entropy loss

    Args:
        predictions (np.array): predictions done by the model.
        y_true (np.array): "true" value, what the model is supposed to predict.

    Returns:
        float: binary_cross_entropy of the model
    """
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(predictions[:, 1]) + (1 - y_true) * np.log(predictions[:, 0]))


def get_accuracy(predictions, y_true):
    """
    Calculates the accuracy of the model.

    Args:
        predictions (np.array): predictions done by the model.
        y_true (np.array): "true" value, what the model is supposed to predict.

    Returns:
        float: accuracy of the model
    """
    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == y_true)
    accuracy = correct_predictions / len(y_true)
    return accuracy


def get_val_loss(network, val_X, val_y, loss_function):
    """
    Calculates the validation loss of the model.

    Args:
        predictions (np.array): predictions done by the model.
        y_true (np.array): "true" value, what the model is supposed to predict.

    Returns:
        float: accuracy of the model
    """
    oh_val_y = one_hot(val_y, 2)

    inputs = val_X
    for layer in network:
        layer.forward(inputs)
        layer.activation.forward(layer.output)
        inputs = layer.activation.output

    val_loss = loss_function.calculate(inputs, oh_val_y)
    return np.mean(val_loss), get_accuracy(inputs, val_y)


def draw_loss(network):
    os.makedirs("visuals", exist_ok=True)
    plt.clf()
    plt.figure(figsize=(10, 6))

    plt.plot(network.train_losses, label="Training loss")
    plt.plot(network.val_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.text(0.05, 0.05, network.params, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
    plt.savefig(f"visuals/model_{network.id}_loss.png")


def draw_accu(network):
    os.makedirs("visuals", exist_ok=True)
    plt.clf()
    plt.figure(figsize=(10, 6))

    plt.plot(network.train_accu, label="Training accuracy")
    plt.plot(network.val_accu, label="Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.text(0.7, 0.2, network.params, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
    plt.savefig(f"visuals/model_{network.id}_accuracy.png")


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
