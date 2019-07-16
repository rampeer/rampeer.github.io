import unittest
from keras.layers import Dense
import numpy as np
from keras.models import Sequential
import keras
import random
import tensorflow as tf
import keras.backend as K


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def create_mlp(dim):
    model = Sequential()
    model.add(Dense(8, input_dim=dim))
    model.add(Dense(1))
    return model


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        fix_seeds(42)
        Xs = np.random.normal(size=(1000, 10))
        Ws = np.random.normal(size=10)
        Ys = np.dot(Xs, Ws) + np.random.normal(size=1000)
        model = create_mlp(10)

        assert(np.abs(Ys.sum() - 36.08286897637723) < 1e-5)
        assert(np.abs(np.array(model.get_weights()[0]).sum() - -2.574954) < 1e-5)
        model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-2),
                      loss=keras.losses.MSE)
        history = model.fit(Xs, Ys, batch_size=10, epochs=10)
        assert(np.abs(np.array(model.get_weights()[0]).sum() - -2.4054692) < 1e-5)


if __name__ == '__main__':
    unittest.main()
