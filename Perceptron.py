
import numpy as np


class Perceptron():

    def __init__(self):

        np.random.seed(2)

        # randomly initialize our weights with mean 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # compute sigmoid nonlinearity

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # convert output of sigmoid function to its derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error *
                                 self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network1 = Perceptron()

    print("Random synaptic weights: ")
    print(neural_network1.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    # training_outputs_1 = np.array([[0, 1, 1, 0]]).T  # X1 is 1
    # training_outputs_2 = np.array([[0, 1, 0, 1]]).T  # X2 is 1
    # training_outputs_3 = np.array([[0, 1, 0, 0]]).T  # AND
    # training_outputs_4 = np.array([[0, 0, 1, 1]]).T #OR
    training_outputs_5 = np.array([[0, 0, 1, 1]]).T  # XOR

    neural_network1.train(training_inputs, training_outputs_5, 20000)

    print("Synaptic weights after training: ")
    print(neural_network1.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))

    print("New situation : input data = ", A, B)
    print("Output data: ")
    print(neural_network1.think(np.array([A, B, "1"])))
