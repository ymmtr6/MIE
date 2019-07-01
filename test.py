import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def show_activation(activation, weights, ylim=(0, 50000)):
    x = np.random.randn(1000, 100)
    node = 100
    hidden = 5
    activations = {}
    for i in range(hidden):
        if i != 0:
            x = activations[i - 1]
        w = weights(node)
        z = np.dot(x, w)
        a = activation(z)
        activations[i] = a

    plt.figure(figsize=(18, 4))
    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")
        (n, bins, _) = plt.hist(a.flatten(), 30, range=(0, 1))
        plt.ylim(ylim)
    plt.show()


show_activation(relu, lambda n: np.random.randn(
    n, n) * np.sqrt(2.0 / n), (0, 7000))
