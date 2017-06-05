import numpy as np


class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.weights = {}
        self.num_layers = 1
        self.adjustments = {}

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def add_layer(self, shape):
        self.weights[self.num_layers] = np.vstack(
            (2 * np.random.random(shape) - 1, 2 * np.random.random((1, shape[1])) - 1))
        self.adjustments[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    def __forward_propagate(self, data):
        activation_values = {}
        activation_values[1] = data
        for layer in range(2, self.num_layers + 1):
            data = np.dot(
                data.T, self.weights[layer - 1][:-1, :]) + self.weights[layer - 1][-1, :].T
            data = self.__sigmoid(data).T
            activation_values[layer] = data
        return activation_values

    def sum_squared_error(self, outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def __back_propagate(self, output, target):
        deltas = {}
        deltas[self.num_layers] = output[self.num_layers] - target

        for layer in reversed(range(2, self.num_layers)):
            a_val = output[layer]
            weights = self.weights[layer][:-1, :]
            prev_deltas = deltas[layer + 1]
            deltas[layer] = np.multiply(
                np.dot(weights, prev_deltas), self.__sigmoid_derivative(a_val))

        for layer in range(1, self.num_layers):
            self.adjustments[layer] += np.dot(deltas[layer + 1],
                                              output[layer].T).T

    def __gradient_descent(self, batch_size, learning_rate):
        for layer in range(1, self.num_layers):
            partial_d = (1 / batch_size) * self.adjustments[layer]
            self.weights[layer][:-1, :] += learning_rate * -partial_d
            self.weights[layer][-1, :] += learning_rate * \
                1e-3 * -partial_d[-1, :]

    def train(self, inputs, targets, epoches, learning_rate=1, stop_accuracy=1e-5):
        error = []
        for iteration in range(epoches):
            for i in range(len(inputs)):
                x = inputs[i]
                y = inputs[i]
                output = self.__forward_propagate(x)
                loss = self.sum_squared_error(output[self.num_layers], y)
                error.append(loss)
                self.__back_propagate(output, y)

            self.__gradient_descent(i, learning_rate)
            if np.mean(error[-(i + 1):]) < stop_accuracy and iteration > 0:
                break

        return (np.asarray(error), iteration + 1)


if __name__ == "__main__":
    nn = NeuralNetwork()

    nn.add_layer((2, 9))
    nn.add_layer((9, 1))

    training_data = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)
    training_labels = np.asarray([[0], [1], [1], [0]])

    error, iteration = nn.train(training_data, training_labels, 5000)
    print('Error = ', np.mean(error[-4:]))
    print('Epoches needed to train = ', iteration)
