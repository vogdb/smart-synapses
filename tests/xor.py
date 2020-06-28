import numpy as np

np.set_printoptions(suppress=True)
np.seterr(all='raise')


def test_klemm(x_train, y_train):
    np.random.seed(666)
    from smart_synapses.klemm.original import Network
    network = Network(x_train, y_train)
    error_hist = network.train(1000)
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
    test_klemm(x_xor, y_xor)
