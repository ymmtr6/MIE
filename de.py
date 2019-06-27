# -*- coding: utf-8 -*-

from scipy.optimize import differential_evolution
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import keras
import json
from pprint import pprint

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

batch_size = 2048
epochs = 1

# lr, beta1, beta2, epsilon, decay
bounds = [(0, 0.5), (0, 0.5), (0.5, 1.0)]


def create_model():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation="relu",
               padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(8, (3, 3), activation="relu",
               padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(
        1, (3, 3), activation="sigmoid", padding="same")(x)
    autoencoder = Model(input_img, decoded)
    return autoencoder


def ae(x):
    #print("lr={}, beta1={}, beta2={}".format(x[0], x[1], x[2]))
    model = create_model()
    adam = keras.optimizers.Adam(
        lr=x[0], beta_1=x[1], beta_2=x[2], epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss="binary_crossentropy")
    model.fit(x_train, x_train, epochs=epochs,
              batch_size=batch_size, shuffle=False,
              validation_split=0.1,
              callbacks=[], verbose=0)
    return model.evaluate(x_test, x_test, verbose=0)


result = differential_evolution(
    ae, bounds, polish=False, disp=True, maxiter=10, updating="deferred", workers=-1)
pprint(result)

with open("output.json", "w") as f:
    json.dump(result, f, indent=4)
