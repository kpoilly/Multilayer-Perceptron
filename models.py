from typing import Type
import numpy as np


class Network:
    """
Class representing the Artificial Neural Network
    """
    def __init__(self):
        self.id = 0
        self.network = []
        self.params = ""
        self.mean = 0
        self.std_dev = 0
        self.train_losses = []
        self.train_accu = []
        self.val_losses = []
        self.val_accu = []
        self.accuracy = 0


class ActivationFunction:
    """
Template class for activation functions
    """
    pass


class LossFunction:
    """
Template class for loss calculation functions
    """
    def calculate(self, output, Y):
        losses = self.forward(output, Y)
        return np.mean(losses)


class DenseLayer:
    """
 class representing a layer of the multilayer perceptron.
    """
    def __init__(self, n_inputs: int, n_neurons: int,
                 activation: Type[ActivationFunction]):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

        # Adam optimizer parameters
        self.m_dw = np.zeros_like(self.weights)
        self.m_db = np.zeros_like(self.biases)
        self.v_dw = np.zeros_like(self.weights)
        self.v_db = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, grad, lr):
        grad = self.activation.backward(grad)
        weights_grad = np.dot(self.inputs.T, grad)
        self.weights -= lr * weights_grad
        self.biases -= lr * np.mean(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

    def adam_backward(self, grad, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, epoch=0):
        grad = self.activation.backward(grad)
        weights_grad = np.dot(self.inputs.T, grad)
        biases_grad = np.mean(grad, axis=0, keepdims=True)

        self.m_dw = beta1 * self.m_dw + (1 - beta1) * weights_grad
        self.m_db = beta1 * self.m_db + (1 - beta1) * biases_grad
        self.v_dw = beta2 * self.v_dw + (1 - beta2) * weights_grad ** 2
        self.v_db = beta2 * self.v_db + (1 - beta2) * biases_grad ** 2

        m_dw_corrected = self.m_dw / (1 - beta1 ** (epoch + 1))
        m_db_corrected = self.m_db / (1 - beta1 ** (epoch + 1))
        v_dw_corrected = self.v_dw / (1 - beta2 ** (epoch + 1))
        v_db_corrected = self.v_db / (1 - beta2 ** (epoch + 1))

        self.weights -= lr * m_dw_corrected / (np.sqrt(v_dw_corrected) + epsilon)
        self.biases -= lr * m_db_corrected / (np.sqrt(v_db_corrected) + epsilon)
        return np.dot(grad, self.weights.T)



class ReLU(ActivationFunction):
    """
 class representing the ReLU activation function.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, grad):
        return grad * (self.inputs > 0)


class Softmax(ActivationFunction):
    """
class representing the Softmax acivation function.
    """
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = prob

    def backward(self, grad):
        return grad


class CrossEntropy(LossFunction):
    """
class representing the CrossEntropy loss calculation function.
    """
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(confidences)

    def backward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            one_hot = np.zeros((len(y_true), y_pred.shape[1]))
            one_hot[range(len(y_true)), y_true] = 1
            y_true = one_hot
        return (y_pred_clipped - y_true) / len(y_pred)
