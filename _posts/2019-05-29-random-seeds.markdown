---
layout: post
title:  "Reproducibility: fixing random seeds not enough"
date:   2019-05-29 00:00:00 +0300
categories: 
---

>"Reproducibile research is easy. Just log your parameters and metrics somewhere, fix seeds, and you are good to go" 

-- me, about two weeks ago. 

Oh boy, I was wrong.

---

There are a lot of workshops, tutorials and conferences on reproducible research. 
Plethora of utilities, tools and frameworks are made in order to help us make nice solutions that can be easily.
However, there are problems and pitfalls that do not appear in simple tutorial project, but bound to happen
in any real research. Very few people actually talk about them. In this post, I'll share a small story about random seeds.

I have a project related to computer vision (handwriting author identification). 

A week ago I managed to achieve decent performance of the identification.
The project ends in July, so I decided to spend time refactoring code and tidying up the project.
I split my large Keras model into several stages, designed test sets for each, 
and used ML Flow to track results of and performance of each stage (it was quite hard - but that's a story for another day). 
So, after a week of refactoring, I have built a nice pipeline, caught a few bugs, managed
to fiddle a bit with hyperparameters and slightly improved performance. 

However, I noticed one peculiarity.
I fixed all random seeds, as numerous guides suggested:

```python
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
```

However, for some reason two consecutive runs with identical hyperparameters gave different results.

Being unable to track the issue down in the project, I decided to make a script with a small model that
reproduces the issue.

```python
import unittest
from keras.layers import Dense
import numpy as np
from keras.models import Sequential
import keras

from .common import fix_seeds

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

```

I ran it several times. It gave same results each execution (I hardcoded them as assertions).

So, I added a bit of complexity into the model by plugging in convolution layers.

```python
import unittest

from .common import fix_seeds

from keras import Input, Model
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
import keras


def create_mlp(dim):
    input = Input(shape=dim)
    conv = Conv2D(5, (3, 3), activation="relu")(input)
    flat = Flatten()(conv)
    output = Dense(1)(flat)
    return Model([input], [output])


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        fix_seeds(42)

        model = create_mlp((20, 20, 3))

        Xs = np.random.normal(size=(1000, 20, 20, 3))
        Ws = np.random.normal(size=(20*20*3, 1))
        Ys = np.dot(Xs.reshape((1000, 20*20*3)), Ws) + np.random.normal(size=(1000, 1))
        assert(np.abs(np.array(model.get_weights()[0]).sum() - -0.96723086) < 1e-7)
        assert(np.abs(Ys.sum() - 418.55143288343953) < 1e-7)

        model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-2),
                      loss=keras.losses.MSE)

        model.fit(Xs, Ys, batch_size=10, epochs=10)

        model_weights = model.get_weights()[0].sum()

        if abs(model_weights - -1.2788088) < 1e-7:
            print("It seems that you are using CPU to train the model!")
        else:
            print("Model weight sum: ", model_weights)


if __name__ == '__main__':
    unittest.main()

```

And, voila, it broke. Every time the script is run, a different number is printed.

Quick research discovered ugly truth: there are [issues with reproducibility](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/).
(to all "you suffer because you use Keras", Pytorch has [similar problem](https://pytorch.org/docs/stable/notes/randomness.html))

Why does this happen?

Some complex operations do not have well-defined order of sub-operations on  the hardware level.
For example, convolution is just a bunch of additions, but the order of these additions is not defined.
So, each execution results in different order of summations. 
And because we operate with floating-point with finite precision, convolution yields slightly different results.

Yep, order of summation matters, `(a+b)+c != a+(b+c)`! You can even check it yourself:

```python
import numpy as np

np.random.seed(42)

xs = np.random.normal(size=10000)
a = 0.0
for x in xs:
    a += x

b = 0
for x in reversed(xs):
    b += x

print(a, b)
print("Difference: ", a - b)
```
Should print
```
-21.359833684261957 -21.359833684262377
Difference: 4.192202140984591e-13
```

Yes, it is "just 4e-13". But because this imprecision happens at each level of deep neural network, and 
for each batch, this error accumulates over layers and time, and model weights diverge significantly.

One might argue that these small discrepancies between runs should not affect the model performance, and
in fact fixing random seed is not that important.

Well, there are merits to this argument. Randomness affects weights; so, model performance 
*depends* on the random seed. But because random seed is not an essential part of the model, 
it might be useful to evaluate model several times for different seeds (or let GPU randomize), 
and report averaged values along with confidence intervals. 
However, very few papers actually do that, and compare models with baseline based on point estimates. 
Instead, there are concerns that improvements that paper report are *less* than this randomness confidence interval.

What's even worse is that results in irreducible stochasticity of the code. 
Imagine a situation. You have found a paper that describes a fancy model - and it has cleanly organized open source 
implementation. "Bingo"! You download the code and the model, start training it. After a couple days (it's *very fancy* model)
you realize that train and validation losses do not decrease. Is it a problem with dataset? 
With hardware or driver versions? Is it a problem on your side, or in the repository itself? Until
everything is fixed and reproducible, you can't be sure.

Nowadays there are plenty of "flimsy" models which tend to get stuck in some local minimum during 
training. Deep RL models are notoriously known for this issue, but in fact any large model have this issue to a degree.

So, it seems that having ability at least *in theory* exactly reproduce model can be extremely handy.

Cherry on the top - because it is a hardware issue, there is no cheap way to fix this.

The only way viable work-around is to define order of operations yourself, i.e. rewrite convolution as a 
series of sum. Obviously, this can considerably slows down the execution, as calling sum operation from application
level adds some overhead in comparison with hardware-optimized call.

Happily, CuDNN already has "reproducible" implementation of these operations, so you do not have to write it yourself.
Enabling a flag should be enough (in theory, again). I'll share details and caveats of this procedure in next post.

Stay tuned!

***

Fixing random seeds is not enough. You have to enable several flags -
and even then, models might give different results depending on hardware, drivers and software versions.

If you ever read a CV paper with relatively small increases of metric and point estimates of the models, take 
it with a pinch of salt ;)

***
