import numpy as np

np.random.seed(666)
np.set_printoptions(suppress=True)
np.seterr(all='raise')


def test_klemm_original(x_train, y_train, epochs):
    from smart_synapses.klemm.original import Network
    network = Network(x_train, y_train)
    error_hist = network.train(epochs)
    print(error_hist)


def test_klemm_experimental(x_train, y_train, epochs):
    from smart_synapses.klemm.experimental import Network
    network = Network(x_train, y_train)
    error_hist = network.train(epochs)
    print(error_hist)


if __name__ == '__main__':
    x_xor = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])
    y_xor = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
    ])
    test_klemm_original(x_xor, y_xor, 200)
    test_klemm_experimental(x_xor, y_xor, 200)
