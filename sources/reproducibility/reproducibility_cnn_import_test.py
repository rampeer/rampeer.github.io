import unittest
import argparse

from .common import assert_same_across_runs


def create_nnet(dim):
    input = Input(shape=dim)
    conv = Conv2D(5, (3, 3), activation="relu")(input)
    pool = MaxPool2D()(conv)
    flat = Flatten()(pool)
    output = Dense(1)(flat)
    return Model([input], [output])


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        fix_seeds(42)

        model = create_nnet((20, 20, 3))

        Xs = np.random.normal(size=(1000, 20, 20, 3))
        Ws = np.random.normal(size=(20*20*3, 1))
        Ys = np.dot(Xs.reshape((1000, 20*20*3)), Ws) + np.random.normal(size=(1000, 1))

        init_weights = np.array(model.get_weights()[0]).sum()

        model.compile(optimizer=RMSprop(lr=1e-2),
                      loss=MSE)

        model.fit(Xs, Ys, batch_size=10, epochs=10)

        model_weights = model.get_weights()[0].sum()

        assert_same_across_runs("mixed model weight before training", init_weights)
        assert_same_across_runs("mixed model weight after training", model_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_before_keras", action="store_true")
    parser.add_argument("--torch_after_keras", action="store_true")
    args = parser.parse_args()

    if args.torch_before_keras:
        print("Torch before keras")
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from keras.losses import MSE
    from keras.optimizers import RMSprop
    from keras import Input, Model
    from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
    import numpy as np
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


    if args.torch_after_keras:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner().run(suite)
