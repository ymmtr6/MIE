# -*- coding: utf-8 -*-
"""

http://programdl.hatenablog.com/entry/2018/11/10/185141
"""

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import optimizers
from keras.utils import plot_model
import numpy as np
from keras import backend as K


class CAE(object):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0):
        self._create_model()

        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.astype("float32") / 255.
        x_test = x_test.astype("float32") / 255.
        self.x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        self.x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay

    def _create_model(self):
        self.input_img = Input(shape=(28, 28, 1))
        x = Conv2D(16, (3, 3), activation="relu",
                   padding="same")(self.input_img)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        self.encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(8, (3, 3), activation="relu",
                   padding="same")(self.encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu")(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Conv2D(
            1, (3, 3), activation="sigmoid", padding="same")(x)

        self.autoencoder = Model(self.input_img, self.decoded)

    def train(self, epochs=10, batch_size=256):
        self.adam = optimizers.Adam(
            self.lr, self.beta_1, self.beta_2, self.epsilon, self.decay)
        self.autoencoder.compile(
            optimizer=self.adam, loss="binary_crossentropy")
        self.autoencoder.fit(self.x_train, self.x_train, epochs=epochs, batch_size=batch_size,
                             shuffle=True, validation_data=(self.x_test, self.x_test), callbacks=[])
        return self.autoencoder.evaluate(self.x_test, self.x_test, verbose=0)


if __name__ == "__main__":
    obj = CAE()
    obj.autoencoder.summary()
    # plot_model(obj.autoencoder, to_file="test.png", show_shapes=True)
    obj.train()
