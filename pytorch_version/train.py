import shutil
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from sklearn.preprocessing import StandardScaler
from models import Network
from utils import load_data, draw_loss_accu, save_network


device = 'cuda' if torch.cuda.is_available() else ' cpu'

def train(network, lr, batch_size, epochs, X, val_X, patience, optimize):
    """
Model training using mini-batchs
    """

    print("Beginning training with following setting:")
    print(f"learning_rate: {lr}")
    print(f"batches of size: {batch_size}")
    print(f"number of epochs: {epochs}")
    print("loss function: CrossEntropy")
    if optimize:
        print("Adam optimizer enabled")
    print("")
    time.sleep(1)

    begin = time.time()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(network.parameters(), lr=lr) if optimize else SGD(network.parameters(), lr=lr)
    
    scaler = StandardScaler()
    norm_X = scaler.fit_transform(X[:, 1:].astype(float))
    norm_val_X = scaler.transform(val_X[:, 1:].astype(float))
    val_y = val_X[:, 0].astype(float)
    
    best_val_loss = float('inf')
    epochs_without_impr = 0

    for epoch in range(epochs):
        batch_indexes = np.random.choice(len(norm_X), batch_size, replace=False)
        batch_X = norm_X[batch_indexes]
        batch_y = X[batch_indexes][:, 0].astype(float)
        
        inputs = torch.tensor(batch_X, dtype=torch.float32).to(device)
        targets = torch.tensor(batch_y, dtype=torch.long).to(device)

        # Forwardpropagation
        outputs = network(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_inputs = torch.tensor(norm_val_X, dtype=torch.float32).to(device)
            val_targets = torch.tensor(val_y, dtype=torch.long).to(device)
            val_outputs = network(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item() 
            val_accu = (torch.argmax(val_outputs, dim=1) == val_targets).float().mean()

        # Early-stopping
        if round(val_loss, 5) < best_val_loss:
            best_val_loss = round(val_loss, 5)
            epochs_without_impr = 0
        else:
            epochs_without_impr += 1

        if epochs_without_impr >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

        network.val_losses.append(val_loss)
        network.val_accu.append(val_accu.item())
        network.train_losses.append(loss.item())
        
        with torch.no_grad():
            train_accu = (torch.argmax(outputs, dim=1) == targets).float().mean()
        network.train_accu.append(train_accu.item())

        print(f"epoch {epoch + 1}/{epochs} - loss: {loss} - val_loss: {val_loss} - Model Accuracy: {round(network.val_accu[-1], 5) * 100}%")

    print(f"\nTraining ended in {round(time.time() - begin, 4)}s.")
    save_network(network)
    draw_loss_accu(network)


def main():
    X = load_data("../data/data_train.csv")
    if X is None:
        print("Error: Cannot found data/data_train.csv\nDid you separate the data file first?", file=sys.stderr)
        return 1
    else:
        print("../data/data_train.csv successfully loaded.")

    val_X = load_data("../data/data_validation.csv")
    if val_X is None:
        print("Error: Cannot found data/data_validation.csv\nDid you separate the data file first?", file=sys.stderr)
        return 1
    else:
        print("../data/data_validation.csv successfully loaded.")

    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--layers', type=int, default=2, choices=range(0, 128),
                        help="Number of layers between input and output layer")
    parser.add_argument('--layersW', type=int, default=16, choices=range(2, 256),
                        help="Number of neurons per layer")
    parser.add_argument('--epochs', type=int, default=10000, choices=range(0, 500001),
                        help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, choices=range(0, len(X) + 1),
                        help="Batch size")
    parser.add_argument('--patience', type=int, default=10, choices=range(0, 201),
                        help="Number of epochs without improvement tolerated (early stopping)")
    parser.add_argument('--clear', action="store_true",
                        help="Delete every models and their data visuals")
    parser.add_argument('--optimize', action="store_true",
                        help="Enable Adam optimizer backpropagation")
    args = parser.parse_args()

    if args.clear:
        try:
            shutil.rmtree("models")
            print("folder models has been deleted.")
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree("visuals")
            print("folder visuals has been deleted.")
        except FileNotFoundError:
            pass
        return 0

    n_inputs = len(X[0]) - 1
    network = Network(n_inputs, args.layers, args.layersW, 2).to(device)
    network.params = f"{args.layers} hidden layers of {args.layersW} neurons\nLearning Rate: {args.lr}\nBatch_Size: {args.batch_size}\nEpochs: {args.epochs}\nPatience: {args.patience}"
    if args.optimize:
        network.params += "\nAdam optimizer enabled"
   
    print("\nNetwork created with following configuration:")
    print(f"Input layer of {args.layersW} neurons, activation function: ReLU")
    print(f"{args.layers} layers of {args.layersW} neurons, activation function: ReLU")
    print("Output layer of 2 neurons, activation function: SoftMax")
    print("")
    time.sleep(1)

    train(network, args.lr, args.batch_size, args.epochs, X, val_X, args.patience, args.optimize)


if __name__ == "__main__":
    main()
