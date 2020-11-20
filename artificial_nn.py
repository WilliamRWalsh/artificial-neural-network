import numpy as np
import random


class ArtificialNeuralNetwork():
    def __init__(self):
        self.alpha = 0.5

        # Training Data
        self.training_X = np.array(
            ([2, 2], [4, 8], [6, 12], [8, 16]), dtype=float)
        self.training_y = np.array(([20], [40], [60], [80]), dtype=float)

        self.training_X = self.training_X / np.amax(self.training_X, axis=0)
        self.training_y = self.training_y / np.amax(self.training_y, axis=0)

        # Network
        self.input_layer_size = 2
        self.hidden_layer_1_size = 3
        self.output_layer_size = 1

        self.W_1 = np.random.randn(
            self.input_layer_size, self.hidden_layer_1_size)
        self.W_2 = np.random\
            .randn(self.hidden_layer_1_size, self.output_layer_size)
        self.b_1 = np.zeros(self.hidden_layer_1_size)
        self.b_2 = np.zeros(self.output_layer_size)

        # Network info
        self.costs = []

    def train(self, iterations):
        for _ in range(iterations):
            self.change_wieghts_and_bias()

    def forward_pass(self):
        self.z_2 = np.dot(self.training_X, self.W_1) + self.b_1
        self.a_2 = self.sigmoid(self.z_2)

        self.z_3 = np.dot(self.a_2, self.W_2) + self.b_2
        self.a_3 = self.sigmoid(self.z_3)

        # y hat
        return self.a_3

    def back_propgation(self):
        self.y_hat = self.forward_pass()

        delta3 = np.multiply(
            -(self.training_y - self.y_hat), self.sigmoid_prime(self.z_3)
        )
        dJdW2 = np.dot(self.a_2.T, delta3)

        delta2 = np.dot(delta3, self.W_2.T) * self.sigmoid(self.z_2)
        dJdW1 = np.dot(self.training_X.T, delta2)

        return dJdW1, dJdW2

    def cost_function(self):
        """
        J = Σ 1/2(y - ŷ)^2
        """
        self.y_hat = self.forward_pass()
        x = 1/2 * (self.training_y - self.y_hat) ** 2
        return x

    def change_wieghts_and_bias(self):
        dJdW1, dJdW2 = self.back_propgation()

        self.W_1 += -self.alpha * dJdW1
        self.W_2 += -self.alpha * dJdW2

    @staticmethod
    def sigmoid(z):
        """
        Activation function
        """
        return 1/(1+np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def test(self, x):
        self.training_X = x / np.amax(x, axis=0)
        print(self.forward_pass())


ann = ArtificialNeuralNetwork()
ann.train(1000)
print(ann.y_hat)
x = np.array(
    ([1, 1], [2, 8], [6, 12], [10, 20]), dtype=float)
ann.test(x)

"""
Next steps:
    - How to normalize input data?
    - Clean up this mess
    - Hook it up to the MNIST data
    - Have the training stop after cost is below threshold?
"""
