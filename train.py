import shutil
import sys
import time
import argparse
import numpy as np
from utils import load_data, normalize_data, normalize_data_spec, one_hot, save_network, get_val_loss, draw_loss, draw_accu, get_accuracy
from models import Network, DenseLayer, ReLU, Softmax, CrossEntropy


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
    loss_function = CrossEntropy()
    norm_X, network.mean, network.std_dev = normalize_data(X[:, 1:].astype(float))

    val_y = val_X[:, 0].astype(float).astype(int)
    val_X = normalize_data_spec(val_X[:, 1:].astype(float), network.mean, network.std_dev)
    best_val_loss = float('inf')
    epochs_without_impr = 0

    for epoch in range(epochs):
        batch_indexes = np.random.choice(len(norm_X), batch_size, replace=False)

        batch_X = norm_X[batch_indexes]
        batch_y = X[batch_indexes][:, 0].astype(float).astype(int)
        oh_batch_y = one_hot(batch_y, 2)

        # Forwardpropagation
        inputs = batch_X
        for layer in network.network:
            layer.forward(inputs)
            layer.activation.forward(layer.output)
            inputs = layer.activation.output

        # Backpropagation
        loss = loss_function.calculate(inputs, oh_batch_y)
        grad = loss_function.backward(inputs, oh_batch_y)

        for layer in reversed(network.network):
            if optimize:
                grad = layer.adam_backward(grad, lr, epoch=epoch)
            else:
                grad = layer.backward(grad, lr)

        # Validation
        val_loss, val_accu = get_val_loss(network.network, val_X, val_y, loss_function)

        if round(val_loss, 5) < best_val_loss:
            best_val_loss = round(val_loss, 5)
            epochs_without_impr = 0
        else:
            epochs_without_impr += 1

        # Early-stopping
        if epochs_without_impr >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

        network.val_losses.append(val_loss)
        network.val_accu.append(val_accu)
        network.train_accu.append(get_accuracy(inputs, batch_y))
        network.train_losses.append(loss)

        print(f"epoch {epoch + 1}/{epochs} - loss: {loss} - val_loss: {val_loss}")

    print(f"\nTraining ended in {time.time() - begin}s.")
    save_network(network)
    draw_loss(network)
    draw_accu(network)


def main():
    X = load_data("data/data_train.csv")
    if X is None:
        print("Error: Cannot found data/data_train.csv\nDid you separate the data file first?", file=sys.stderr)
        return 1
    else:
        print("data/data_train.csv successfully loaded.")

    val_X = load_data("data/data_validation.csv")
    if val_X is None:
        print("Error: Cannot found data/data_validation.csv\nDid you separate the data file first?", file=sys.stderr)
        return 1
    else:
        print("data/data_validation.csv successfully loaded.")

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

    network = Network()
    network.params = f"{args.layers} hidden layers of {args.layersW} neurons\nLearning Rate: {args.lr}\nBatch_Size: {args.batch_size}\nEpochs: {args.epochs}\nPatience: {args.patience}"
    if args.optimize:
        network.params += "\nAdam optimizer enabled"
    network.network = [DenseLayer(n_inputs=len(X[0])-1, n_neurons=args.layersW, activation=ReLU())] # input layer
    for i in range(0, args.layers):
        network.network.append(DenseLayer(n_inputs=args.layersW, n_neurons=args.layersW, activation=ReLU()))
    network.network.append(DenseLayer(n_inputs=args.layersW, n_neurons=2, activation=Softmax())) # output layer

    print("\nNetwork created with following configuration:")
    print(f"Input layer of {args.layersW} neurons, activation function: ReLU")
    print(f"{args.layers} layers of {args.layersW} neurons, activation function: ReLU")
    print("Output layer of 2 neurons, activation function: SoftMax")
    print("")
    time.sleep(1)

    train(network, args.lr, args.batch_size, args.epochs, X, val_X, args.patience, args.optimize)


if __name__ == "__main__":
    main()
