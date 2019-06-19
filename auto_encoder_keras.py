# -*- coding: utf-8 -*-
"""

http://programdl.hatenablog.com/entry/2018/11/10/185141
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot

encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(784, activation="sigmoid")(encoded)
ae = Model(input_img, decoded)

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = ae.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

ae.compile(optimizer="adadelta", loss="binary_crossentropy")
