"""
Smart Synapse mechanism from 14.2 Klemm(2000)

Some questions to the article:
 - why delta_penalty is so big in the article? It is 1. It is huge.
 - what to do with memory if it hits the boundaries? Should we reset it? If not
    then how prevent overtraining/overfitting?
 - what to do with weights if they are growing too big? How to prevent that?

"""
import numpy as np

from smart_synapses.utils import get_shape_size


def winner_take_all(neurons, beta_noise):
    """General purpose Winner Take All mechanism.

    It will work with any shaped input.

    Args:
        neurons (numpy.ndarray): neurons input
        beta_noise: beta noise according to the original article

    Returns:
        numpy.ndarray: array of the same shape as input where all elements are 0
        except the winner one. It has 1 in its place.
    """
    shape = neurons.shape
    neurons = np.clip(np.ravel(neurons), -40, 40)
    divisor = np.sum(np.exp(beta_noise * neurons))
    # p - probabilities
    p = np.exp(beta_noise * neurons) / divisor
    winner_flat_idx = np.random.choice(np.arange(len(p)), p=p)
    winner_idx = np.unravel_index(winner_flat_idx, shape)
    winner = np.zeros(shape, dtype=np.int)
    winner[winner_idx] = 1
    return winner


class Layer:
    """Single layer of neurons with smart synapses."""
    def __init__(self, name, shape, theta_threshold):
        self.name = name
        self.input_weights = None
        self.input_memory = None
        self.shape = shape
        self.theta_threshold = theta_threshold

    def connect_input(self, input_size):
        weights_shape = (get_shape_size(self.shape), input_size)
        self.input_weights = np.random.randn(*weights_shape)
        self.reset_memory()

    def reset_memory(self, indices=None):
        if indices is None:
            self.input_memory = np.zeros(self.input_weights.shape)
        else:
            self.input_memory[indices] = 0

    def train(self, input_sample, reward, delta_penalty, beta_noise):
        output_sample = self.input_weights.dot(input_sample)
        output_winner = winner_take_all(output_sample, beta_noise)

        input_active_indices = np.nonzero(input_sample)[0]
        output_active_indices = np.nonzero(output_winner)[0]
        weight_hebb_indices = np.s_[output_active_indices, input_active_indices]
        self.input_memory[weight_hebb_indices] -= reward
        self.input_memory = np.clip(self.input_memory, 0, self.theta_threshold)

        theta_indices = np.nonzero(self.input_memory == self.theta_threshold)
        self.input_weights[theta_indices] -= delta_penalty
        self.reset_memory(theta_indices)

        return output_sample


class Network:
    """Network of layers of neurons with smart synapses."""
    def __init__(self, x_train, y_train, beta_noise=10, delta_penalty=0.01, theta_threshold=2):
        self.x_train = x_train
        self.y_train = y_train
        x_len, *x_shape = x_train.shape
        y_len, *y_shape = y_train.shape
        assert x_len == y_len, '`x_train` and `y_train` must be the same length'
        assert 1 <= len(x_shape) <= 2, 'for now we process input of 1D, 2D only'
        assert 1 <= len(y_shape) <= 2, 'for now we process output of 1D, 2D only'

        self.beta_noise = beta_noise
        self.delta_penalty = delta_penalty
        self.theta_threshold = theta_threshold

        self.layers = [
            # 10 is just a random pick here
            Layer('L1', tuple([3] * len(x_shape)), theta_threshold),
            Layer('output', y_shape, theta_threshold)
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
                y_pred = winner_take_all(self.predict(x), self.beta_noise)
                reward = 1 if np.array_equal(y, y_pred) else -1
                x_l = x
                for layer in self.layers:
                    x_l = layer.train(x_l, reward, self.delta_penalty, self.beta_noise)
            error_hist.append(self.error(self.x_train, self.y_train))
        return error_hist
