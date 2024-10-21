# File: backpropagation.py
import numpy as np
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the weights and biases
W1 = np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]])
W2 = np.array([[0.7], [0.8], [0.9]])
b1 = np.array([0.1, 0.2, 0.3])
b2 = np.array([0.4])

def forward_propagate(X):
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    return output_layer

def backpropagate(X, y_true, learning_rate):
    global W1, W2, b1, b2

    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

    output_error = y_true - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)

    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    X = X.reshape(1, -1)
    W1 += learning_rate * X.T.dot(hidden_delta.reshape(1, -1))
    W2 += learning_rate * hidden_layer.reshape(-1, 1).dot(output_delta.reshape(1, -1))
    b1 += learning_rate * hidden_delta
    b2 += learning_rate * output_delta
