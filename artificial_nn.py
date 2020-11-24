import numpy as np
from keras.datasets import mnist


class ArtificialNeuralNetwork():
    '''
    Simple Artificial Neural Network to classify handwritten digits.
    '''

    def __init__(self, alpha):
        self.alpha = alpha

        # Training & Testing Data
        num_samples = 10
        (self.training_X, self.training_y), (self.test_X,
                                             self.test_y) = mnist.load_data()
        self.training_X = self.training_X[:num_samples]
        self.training_y = self.training_y[:num_samples].reshape(num_samples, 1)
        # TODO: Try using rivel instead of flatten
        self.training_X = np.asarray([x.flatten() for x in self.training_X])

        # Normalize inputs/outputs
        self.training_X = self.training_X / 255
        self.training_y = self.training_y / 9

        # Network
        self.input_layer_size = 28 * 28
        self.hidden_layer_1_size = 64
        self.output_layer_size = 1

        # Weights & Biases
        self.W_1 = np.random.randn(
            self.input_layer_size, self.hidden_layer_1_size)
        self.W_2 = np.random\
            .randn(self.hidden_layer_1_size, self.output_layer_size)
        self.b_1 = np.zeros(self.hidden_layer_1_size)
        self.b_2 = np.zeros(self.output_layer_size)

    def train(self, iterations):
        '''
        Loops over the training data `iterations`# of times updating
        the weights and biases each time.

        Each loop goes:
            - Forward pass
            - Back Propagation
            - Change Weights
        '''
        for _ in range(iterations):
            neurons_io = self.forward_pass()
            dJW1, dJW2 = self.back_propgation(neurons_io)
            self.change_weights_and_bias(dJW1, dJW2)

        # Logs
        print("y = " + str(self.training_y[:10]))
        print("y_hat = " + str(neurons_io['y_hat'][:10]))
        print("Cost = " + str(sum(self.cost_function())))

    def forward_pass(self):
        '''
        ŷ = f(X * W)

        :return dict: Each neurons inputs/outputs
        '''
        z_2 = np.dot(self.training_X, self.W_1) + self.b_1
        a_2 = self.sigmoid(z_2)

        z_3 = np.dot(a_2, self.W_2) + self.b_2
        a_3 = self.sigmoid(z_3)

        return {'z_2': z_2, 'z_3': z_3, 'a_2': a_2, 'y_hat': a_3}

    def back_propgation(self, neurons_io):
        '''
        dJ/dW2 = aT d3
        dJ/dW1 = XT d2
        '''
        delta3 = np.multiply(
            -(self.training_y - neurons_io['y_hat']
              ), self.sigmoid_prime(neurons_io['z_3'])
        )
        dJdW2 = np.dot(neurons_io['a_2'].T, delta3)

        delta2 = np.dot(delta3, self.W_2.T) * \
            self.sigmoid_prime(neurons_io['z_2'])
        dJdW1 = np.dot(self.training_X.T, delta2)

        return dJdW1, dJdW2

    def change_weights_and_bias(self, dJW1, dJW2):
        '''
        Update weights and biases
        '''
        # TODO: Add baises
        self.W_1 += -self.alpha * dJW1
        self.W_2 += -self.alpha * dJW2

    def cost_function(self):
        '''
        J = Σ 1/2(y - ŷ)^2
        '''
        io = self.forward_pass()
        x = 1/2 * (self.training_y - io['y_hat']) ** 2
        return x

    @staticmethod
    def sigmoid(z):
        '''
        Activation function -> f(z)
        '''
        return 1/(1+np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        '''
        Activation function dervitive -> f'(z)
        '''
        return np.exp(-z)/((1+np.exp(-z))**2)

    def test(self, x):
        self.training_X = x / np.amax(x, axis=0)


'''
Main
'''
ann = ArtificialNeuralNetwork(alpha=1)
ann.train(10000)

"""
Next steps:
    - Have the training stop after cost is below threshold?
    - Make 10 output nodes
    - Make extra hidden layers
    - Add bais
    
"""
