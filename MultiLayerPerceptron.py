
import numpy as np


class MLP():

    def __init__(self):

        np.random.seed(2)

        # randomly initialize our weights with mean 0
        self.synaptic_weights_0 = 2 * np.random.random((3, 4)) - 1
        self.synaptic_weights_1 = 2 * np.random.random((4, 1)) - 1

        self.prev_synapse_0_weight_update = np.zeros_like(
            self.synaptic_weights_0)
        self.prev_synapse_1_weight_update = np.zeros_like(
            self.synaptic_weights_1)

        self.synapse_0_direction_count = np.zeros_like(self.synaptic_weights_0)
        self.synapse_1_direction_count = np.zeros_like(self.synaptic_weights_1)

    # compute sigmoid nonlinearity
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # convert output of sigmoid function to its derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, alpha, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            layer_1_output, layer_2_output = self.think(training_inputs)

            # how much did we miss the target value?
            layer_2_error = training_outputs - layer_2_output

            if (iteration % 10000) == 0:
                print("Error after "+str(iteration)+" iterations:" +
                      str(np.mean(np.abs(layer_2_error))))

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * \
                self.sigmoid_derivative(layer_2_output)

            # how much did each l1 value contribute to the l2 error
            # (according to the weights)?
            layer_1_error = np.dot(layer_2_delta, self.synaptic_weights_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * \
                self.sigmoid_derivative(layer_1_output)

            synaptic_weights_1_adjustments = np.dot(
                layer_1_output.T, layer_2_delta)
            synaptic_weights_0_adjustments = np.dot(
                training_inputs.T, layer_1_delta)

            if(iteration > 0):
                self.synapse_0_direction_count += np.abs(
                    ((synaptic_weights_0_adjustments > 0)+0) - ((self.prev_synapse_0_weight_update > 0) + 0))
                self.synapse_1_direction_count += np.abs(
                    ((synaptic_weights_1_adjustments > 0)+0) - ((self.prev_synapse_1_weight_update > 0) + 0))

                self.synaptic_weights_1 += alpha * synaptic_weights_1_adjustments
                self.synaptic_weights_0 += alpha * synaptic_weights_0_adjustments

                self.prev_synapse_0_weight_update = synaptic_weights_0_adjustments
                self.prev_synapse_1_weight_update = synaptic_weights_1_adjustments

    def think(self, inputs):

        # Feed forward through layers 0, 1, and 2
        layer_0 = inputs.astype(float)
        layer_1 = self.sigmoid(np.dot(layer_0, self.synaptic_weights_0))
        layer_2 = self.sigmoid(np.dot(layer_1, self.synaptic_weights_1))

        return layer_1, layer_2


if __name__ == "__main__":

    neural_network1 = MLP()

    print("Random synaptic weights for input layer: ")
    print(neural_network1.synaptic_weights_0)
    print("Random synaptic weights for hidden layer: ")
    print(neural_network1.synaptic_weights_1)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    # training_outputs_1 = np.array([[0, 1, 1, 0]]).T  # X1 is 1
    # training_outputs_2 = np.array([[0, 1, 0, 1]]).T  # X2 is 1
    # training_outputs_3 = np.array([[0, 1, 0, 0]]).T #AND
    # training_outputs_4 = np.array([[0, 0, 1, 1]]).T #OR
    training_outputs_5 = np.array([[0, 0, 1, 1]]).T  # XOR
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for alpha in alphas:
        print("\nTraining With Alpha:" + str(alpha))
        neural_network1.train(alpha, training_inputs,
                              training_outputs_5, 20000)
        print("Updated synaptic weights for input layer: ")
        print(neural_network1.synaptic_weights_0)

        print("Input layer synaptic weights update direction changes")
        print(neural_network1.synapse_0_direction_count)

        print("Updated synaptic weights for hidden layer: ")
        print(neural_network1.synaptic_weights_1)

        print("Hidden layer synaptic weights update direction changes")
        print(neural_network1.synapse_1_direction_count)

        A = str(input("Input 1: "))
        B = str(input("Input 2: "))

        print("New situation : input data = ", A, B)
        print("Output data: ")
        hidden, output = neural_network1.think(np.array([A, B, "1"]))
        print(output)
