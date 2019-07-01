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
import argparse
import os
import json_encoder
import warnings

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description="DE-heyperParam")

parser.add_argument("-o", "--output", default="output.json")
parser.add_argument("-w", "--workers", default=-1)
parser.add_argument("-m", "--maxiter", default=10)
parser.add_argument("-p", "--popsize", default=5)
parser.add_argument("-e", "--epochs", default=1)
parser.add_argument("-b", "--batch_size", default=1024)
parser.add_argument("-v", "--verbose", default=0)
args = parser.parse_args()

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

batch_size = args.batch_size
epochs = args.epochs

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
    # print("lr={}, beta1={}, beta2={}".format(x[0], x[1], x[2]))
    model = create_model()
    model._make_predict_function()
    adam = keras.optimizers.Adam(
        lr=x[0], beta_1=x[1], beta_2=x[2], epsilon=1e-07, decay=0.0)
    model.compile(optimizer=adam, loss="binary_crossentropy")
    keras.callbacks.EarlyStopping()
    model.fit(x_train, x_train, epochs=epochs,
              batch_size=batch_size, shuffle=False,
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping()], verbose=args.verbose)
    return model.evaluate(x_test, x_test, verbose=0)


"""
differential_evolution

ARGS:
    - updating: 個体の更新を行うタイミングを指定する.
    "immediate", "deferred"の二種類があるが、並列計算する場合は"deferred"
    - workers: 並列に計算を行う。並列プロセス数の指定。-1を指定した場合はCPUのコア数。

{'fun': 0.22151452987194062,
 'message': 'Maximum number of iterations has been exceeded.',
 'nfev': 165,
 'nit': 10,
 'success': False,
 'x': array([0.00612182, 0.48125592, 0.79976852])}

"""
result = differential_evolution(
    ae, bounds, polish=False, disp=True, maxiter=10, updating="deferred", workers=-1, popsize=5)
pprint(result)

with open(args.output, "w") as f:
    json.dump(result, f, indent=4, cls=json_encoder.MyEncoder)
