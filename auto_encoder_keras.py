# -*- coding: utf-8 -*-
"""

http://programdl.hatenablog.com/entry/2018/11/10/185141
"""

import keras
from keras.models import load_model
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

"""
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_valid = train_test_split(x_train, test_size=0.175)
x_train, x_train.astype("float32") / 255.
x_valid = x_valid.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = x_train.reshape((len(x_trian), np.prod(x_train.shape[1:])))
"""
