import numpy as np


class ArtificialNeuralNetwork():
    def __init__(self, alpha):
        self.alpha = alpha

        # Training Data
        self.training_X = np.array(
            ([2, 2], [4, 4], [6, 6], [8, 8]), dtype=float)
        self.training_y = np.array(([20], [40], [60], [80]), dtype=float)

        self.training_X = self.training_X / np.amax(self.training_X, axis=0)
        self.training_y = self.training_y / 100

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

    def train(self, iterations):
        for _ in range(iterations):
            neurons_io = self.forward_pass()
            dJW1, dJW2 = self.back_propgation(neurons_io)
            self.change_wieghts_and_bias(dJW1, dJW2)
            print(neurons_io['y_hat'])

    def forward_pass(self):
        z_2 = np.dot(self.training_X, self.W_1) + self.b_1
        a_2 = self.sigmoid(z_2)

        z_3 = np.dot(a_2, self.W_2) + self.b_2
        a_3 = self.sigmoid(z_3)

        return {'z_2': z_2, 'z_3': z_3, 'a_2': a_2, 'y_hat': a_3}

    def back_propgation(self, neurons_io):
        delta3 = np.multiply(
            -(self.training_y - neurons_io['y_hat']
              ), self.sigmoid_prime(neurons_io['z_3'])
        )
        dJdW2 = np.dot(neurons_io['a_2'].T, delta3)

        delta2 = np.dot(delta3, self.W_2.T) * \
            self.sigmoid_prime(neurons_io['z_2'])
        dJdW1 = np.dot(self.training_X.T, delta2)

        return dJdW1, dJdW2

    def change_wieghts_and_bias(self, dJW1, dJW2):
        self.W_1 += -self.alpha * dJW1
        self.W_2 += -self.alpha * dJW2

    def cost_function(self):
        """
        J = Σ 1/2(y - ŷ)^2
        """
        self.y_hat = self.forward_pass()
        x = 1/2 * (self.training_y - self.y_hat) ** 2
        return x

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
        print(self.forward_pass()['y_hat'])


ann = ArtificialNeuralNetwork(alpha=1)
ann.train(10000)

# x = np.array(
#     ([1, 1], [2, 8], [6, 12], [10, 20]), dtype=float)
# ann.test(x)

"""
Next steps:
    - How to normalize input data?
    - Hook it up to the MNIST data
    - Have the training stop after cost is below threshold?
"""
