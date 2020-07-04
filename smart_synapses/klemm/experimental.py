"""
Just some experiments with the original version
"""
import numpy as np

from smart_synapses.utils import get_shape_size, winner_take_all


class Layer:
    def __init__(self, name, shape, memory_threshold):
        self.name = name
        self.input_weights = None
        self.input_memory = None
        # Try to actually create neurons here and use them in `predict`. Make some filter from them
        # like real neurons. self.neurons = np.random.randn(*shape). Those neurons can take random
        # background values like background spiking activity.
        self.shape = shape
        self.memory_threshold = memory_threshold

    def connect_input(self, input_size):
        weights_shape = (get_shape_size(self.shape), input_size)
        self.input_weights = np.random.randn(*weights_shape)
        self.reset_memory()

    def reset_memory(self, indices=None):
        if indices is None:
            self.input_memory = .5 * self.memory_threshold * np.zeros(self.input_weights.shape)
        else:
            self.input_memory[indices] = 0

    def train(self, input_sample, reward, weight_penalty):
        output_sample = self.input_weights.dot(input_sample)
        output_winner = winner_take_all(output_sample)

        input_active_indices = np.nonzero(input_sample)[0]
        output_active_indices = np.nonzero(output_winner)[0]
        weight_hebb_indices = np.s_[output_active_indices, input_active_indices]
        # possibly do `reward * output_winner` when gaussian som
        self.input_memory[weight_hebb_indices] += reward
        self.input_memory = np.clip(self.input_memory, 0, self.memory_threshold)

        strengthen_indices = np.nonzero(self.input_memory == self.memory_threshold)
        self.input_weights[strengthen_indices] += weight_penalty
        self.reset_memory(strengthen_indices)
        weaken_indices = np.nonzero(self.input_memory == 0)
        self.input_weights[weaken_indices] -= weight_penalty
        self.reset_memory(weaken_indices)

        return output_sample


class Network:

    def __init__(self, x_train, y_train, weight_penalty=0.01, memory_threshold=2):
        # Apply some random background noise to neurons? Both during train and test.
        self.x_train = x_train
        self.y_train = y_train
        x_len, *x_shape = x_train.shape
        y_len, *y_shape = y_train.shape
        assert x_len == y_len, '`x_train` and `y_train` must be the same length'
        assert 1 <= len(x_shape) <= 2, 'for now we process input of 1D, 2D only'
        assert 1 <= len(y_shape) <= 2, 'for now we process output of 1D, 2D only'

        self.weight_penalty = weight_penalty
        self.memory_threshold = memory_threshold

        self.layers = [
            # 10 is just a random pick here
            Layer('L1', tuple([10] * len(x_shape)), memory_threshold),
            Layer('output', y_shape, memory_threshold)
        ]
        self.layers[0].connect_input(get_shape_size(x_shape))
        for idx, layer in enumerate(self.layers[1:]):
            layer.connect_input(get_shape_size(self.layers[idx].shape))

    def predict(self, x):
        result = self.layers[0].input_weights.dot(x.T)
        for layer in self.layers[1:]:
            result = layer.input_weights.dot(result)
        return result.T

    def error(self, x, y_true):
        y_predict = self.predict(x)
        return np.sum(np.abs(y_predict - y_true))

    def train(self, epochs):
        error_hist = []
        for epoch in range(epochs):
            for (x, y) in zip(self.x_train, self.y_train):
                # better `y_pred` adn error calculation here. We should not do WTA on it probably?
                # how do WTA if it is gaussion som?
                y_pred = winner_take_all(self.predict(x))
                reward = 1 if np.array_equal(y, y_pred) else -1
                x_l = x
                for layer in self.layers:
                    x_l = layer.train(x_l, reward, self.weight_penalty)
            error_hist.append(self.error(self.x_train, self.y_train))
        return error_hist
