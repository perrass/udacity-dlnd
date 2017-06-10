import numpy as np


class Activation(object):


class Loss(object):


class Optimizer(object):


class NeuralNetwork(object):
    def __init__(self):
        """
        weights: the last row is bias, and the others are weights
        """
        self.weights = {}
        self.num_layers = 1
        self.layers_shape = {}
        self.activation = Activation()
        self.loss = Loss()
        self.optimizer = Optimizer()

    def add_layer(self, shape):
        self.weights[self.num_layers] = np.vstack(
            (np.random.random(shape) - 0.5, np.random.random((1, shape[1])) - 0.5))
        self.num_layers += 1

    def train(self, inputs, targets, batch_size, epoches, learning_rate=0.01, tolerance=1e-5):
        num_batches = int(len(inputs) / batch_size)
        for epoch in epoches:
            for i in range(num_batches):
                X = inputs[i:i + batch_size]
                y = targets[i:i + batch_size]
                outputs = self.__forward_propagate(x)
                partial = self.__back_propagate(outputs, y)
                loss = self.loss()
            self.optimizer()

    def predict(self, inputs):


if __name__ == "__main__":
    # Create instance of a neural network
    nn = NeuralNetwork()

    # Add Layers (Input layer is created by default)
    nn.add_layer((2, 9))
    nn.add_layer((9, 1))

    # XOR function
    training_data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_labels = np.asarray([[0], [1], [1], [0]])

    error, iteration = nn.train(training_data, training_labels, 5000)
    print('Error = ', np.mean(error[-4:]))
    print('Epoches needed to train = ', iteration)
