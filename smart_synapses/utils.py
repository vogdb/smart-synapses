from functools import partial
import numpy as np
from scipy.ndimage import gaussian_filter


def get_shape_size(shape):
    """Gets number of items that fit into an array of ``shape``."""
    return np.prod(shape)


def gaussion_som(sigma, truncate):
    """Gaussian filter function for Self Organizing Map of neurons.

    Args:
      sigma: sigma for ``gaussian_filter``
      truncate: truncate for ``gaussian_filter``

    Returns:
      Function: ``gaussian_filter`` partially bound with given ``sigma``, ``truncate``
    """
    return partial(gaussian_filter, sigma=sigma, truncate=truncate)


def winner_take_all_prob(neurons, som=None):
    """General purpose Winner Take All mechanism.

    It will work with any shaped input. Neurons values converted to probabilities and then the
    winner is defined by ``np.random.choice`` among those probabilities.

    Args:
        neurons (numpy.ndarray): neurons input
        som (Function): Self Organizing Map function. See ``gaussion_som`` for example.
            Function to spread the winner value among its neighbours.

    Returns:
        numpy.ndarray: array of the same shape as ``neurons`` where only winner values are kept.
        Single winner value in case no ``som`` was provided.
    """
    shape = neurons.shape
    clipped_neurons = np.clip(np.ravel(neurons), -50, 50)
    divisor = np.sum(np.exp(clipped_neurons))
    # p - probabilities
    p = np.exp(clipped_neurons) / divisor
    winner_flat_idx = np.random.choice(np.arange(len(p)), p=p)
    winner_idx = np.unravel_index(winner_flat_idx, shape)
    winner = np.zeros(shape)
    winner[winner_idx] = neurons[winner_idx]
    if som is None:
        return winner
    return som(winner)


def winner_take_all(neurons, som=None):
    """General purpose Winner Take All mechanism.

    Filters ``neurons`` to keep only the neuron with the highest value. No probabilities.

    Args:
        neurons (numpy.ndarray): neurons values
        som (Function): Self Organizing Map function. See ``gaussion_som`` for example.
            Function to spread the winner value among its neighbours.

    Returns:
        numpy.ndarray: array of the same shape as ``neurons`` where only winner values are kept.
        Single winner value in case no ``som`` was provided.
    """
    winner_idx = np.unravel_index(neurons.argmax(), neurons.shape)
    result = np.zeros(neurons.shape)
    result[winner_idx] = neurons[winner_idx]
    if som is None:
        return result
    return som(result)
