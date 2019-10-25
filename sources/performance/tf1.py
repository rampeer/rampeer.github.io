import unittest
from time import time

import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.losses import MSE
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib


def log(where, what):
    with open(where, "a") as f:
        f.write(what)


class gauge:
    def __init__(self, file, section):
        self.section = section
        self.file = file

    def __enter__(self):
        self.t = time()

    def __exit__(self, type, value, traceback):
        log(self.file, self.section + " " + str(time() - self.t) + "s\n")


def create_nnet(dim, layers=[(5, 3), (5, 3), (5, 3), (5, 3)]):
    input = Input(shape=dim)
    current_layer = input
    for filters, kernel in layers:
        current_layer = Conv2D(filters, (kernel, kernel), activation="relu", padding="same")(current_layer)
        current_layer = MaxPool2D()(current_layer)
    flat = Flatten()(current_layer)
    output = Dense(1)(flat)
    return Model([input], [output])


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        N = 400
        Ys = np.random.normal(size=N)
        Xs = np.random.normal(size=(N, 256, 256, 3)) + Ys[:, np.newaxis, np.newaxis, np.newaxis] / 10.0
        model = create_nnet((256, 256, 3))
        print(model.summary())

        log("tf1", "TF version:" + tf.__version__)
        log("tf1", "Keras version:" + keras.__version__)
        if K.tensorflow_backend._get_available_gpus():
            log("tf1", "Devices: " + "; ".join([x.physical_device_desc for x in device_lib.list_local_devices()]) + "\n")
        else:
            log("tf1", "No GPU found")

        with gauge("tf1", "training"):
            model.compile(optimizer=SGD(lr=1e-2), loss=MSE)
            model.fit(Xs, Ys, batch_size=16, epochs=30, verbose=0)
        with gauge("tf1", "inference"):
            for _ in range(30):
                model.predict(Xs, batch_size=16)


if __name__ == '__main__':
    unittest.main()
