import unittest
from keras.layers import Dense
import numpy as np
from keras.models import Sequential
import keras
import random
import tensorflow as tf
import keras.backend as K

from common import assert_same_across_runs


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

        init_weights = np.array(model.get_weights()[0]).sum()

        model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-2),
                      loss=keras.losses.MSE)
        model.fit(Xs, Ys, batch_size=10, epochs=10)

        assert_same_across_runs("dense model data", Ys.sum())
        assert_same_across_runs("dense model weight after training", init_weights)
        assert_same_across_runs("dense model weight after training", np.array(model.get_weights()[0]).sum())


if __name__ == '__main__':
    unittest.main()
