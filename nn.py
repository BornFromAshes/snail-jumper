import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i + 1], layer_sizes[i]) * 2 - 1)
            self.biases.append(np.zeros((layer_sizes[i + 1], 1)))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        hidden_layers_input = []
        hidden_layers_output = []
        hidden_layers_input.append(np.dot(self.weights[0], x) + self.biases[0])
        hidden_layers_output.append(self.activation(hidden_layers_input[0]))
        for i in range(len(self.sizes) - 2):
            hidden_layers_input.append(np.dot(self.weights[i + 1], hidden_layers_output[i]) + self.biases[i + 1])
            hidden_layers_output.append(self.activation(hidden_layers_input[i + 1]))
        return hidden_layers_output[len(self.sizes) - 2]
