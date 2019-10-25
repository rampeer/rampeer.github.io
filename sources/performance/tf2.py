import unittest
from time import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
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


def create_nnet(dim, layers_configuration=[(5, 3), (5, 3), (5, 3), (5, 3)]):
    input = keras.Input(shape=dim)
    current_layer = input
    for filters, kernel in layers_configuration:
        current_layer = layers.Conv2D(filters, (kernel, kernel), activation="relu", padding="same")(current_layer)
        current_layer = layers.MaxPool2D()(current_layer)
    flat = layers.Flatten()(current_layer)
    output = layers.Dense(1)(flat)
    return keras.Model([input], [output])


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        N = 400
        Ys = np.random.normal(size=N)
        Xs = np.random.normal(size=(N, 256, 256, 3)) + Ys[:, np.newaxis, np.newaxis, np.newaxis] / 10.0
        model = create_nnet((256, 256, 3))

        log("tf2", "TF version:" + tf.__version__ + "\n")
        log("tf2", "Devices: " + "; ".join([x.physical_device_desc for x in device_lib.list_local_devices()]) + "\n")

        with gauge("tf2", "training"):
            model.compile(optimizer=keras.optimizers.SGD(lr=1e-2), loss=keras.losses.MSE)
            model.fit(Xs, Ys, batch_size=16, epochs=30, verbose=0)
        with gauge("tf2", "inference"):
            for _ in range(30):
                model.predict(Xs, batch_size=16)


if __name__ == '__main__':
    unittest.main()
