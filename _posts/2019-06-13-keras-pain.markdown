---
layout: post
title:  "Practical issues of reproducibility"
date:   2019-05-29 00:00:00 +0300
categories: 
---

We are on the quest towards reproducibility.

As I described in [previous post]({% post_url 2019-05-29-random-seeds %}), fixing random seeds is not enough because
parallelization also throws sticks in the wheels. Because order of operations is not well-defined, we may end up
with slighly different results.

To be specific, it's issue with library that interprets our high-level neural network description into low-level 
GPU commands, namely - CuDNN.

Overall pipeline looks like a layered pie:

```
Keras/Pytorch - top-level library you are usually interacting with
Tensorflow/THNN - back-end for these libraries
CuDNN - extention of CUDA for deep learning
CUDA - platform for parallel computations using GPU
Drivers - hardware-specific piece of software that is needed for universal GPU API
Hardware - your RTX 2080 Ti
```

Thankfully, there are switches in CuDNN that enable and disable deterministic (but slower) implementation
that that produces same results every run. Because we do not interact with CuDNN directly, we have to
tell our library of choice to turn this switch on.

Unfortunately, Keras does not have that functionality (yet), as described in 
[these](https://github.com/tensorflow/tensorflow/issues/18096) 
[issues](https://github.com/tensorflow/tensorflow/issues/12871).

Looks like it's time for Pytorch to shine. It has settings that tell CuDNN to use deterministic implementation:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Let's write a simple network with a single convolution, and train it on random data (exact architecture or data do not
matter much, as we are just testing reproducibility).

```python
import random
import unittest

import numpy as np

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, in_shape: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.hidden_size = int((in_shape - 2) * (in_shape - 2) / 4) * 5
        self.fc1 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.hidden_size)
        x = F.relu(self.fc1(x))
        return x

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        def get_weight_sums():
            return np.sum([np.sum(x.data.cpu().detach().numpy()) for x in net.parameters()])

        fix_seeds(42)

        Ws = np.random.normal(size=(20*20*3, 1))

        net = Net(in_shape=20).cuda()

        if np.abs(get_weight_sums() - -2.2693140506744385) > 1e-7:
            raise Exception(f"Model weight sum after initialization is wrong! It should not be {get_weight_sums()}")

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        for _ in range(1000):
            optimizer.zero_grad()
            Xs = np.random.normal(size=(10, 3, 20, 20))
            Ys = np.dot(Xs.reshape((10, 20 * 20 * 3)), Ws) + np.random.normal(size=(10, 1))

            output = net(torch.tensor(Xs, dtype=torch.float).cuda())
            loss = nn.MSELoss()(output, torch.tensor(Ys, dtype=torch.float).cuda())

            loss.backward()
            optimizer.step()

        if np.abs(get_weight_sums() - -17.0853214263916) > 1e-7:
            raise Exception(f"Model weight sum after training is wrong! It should not be {get_weight_sums()}")


if __name__ == '__main__':
    unittest.main()

```

It seems that these flags work. In fact, for me, setting `benchmark = False` is enough to get consistent results.

The script gives "OK" each time it is run, which means pytorch gives consistent results.

But wait! There are speculations that these variables affect some session-wide variables, so setting them in pytorch
will affect Keras. This way, we can make use of these flags AND use Keras to write models. Let's check it out.

```python
import unittest
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_before_keras", action="store_true")
    parser.add_argument("--torch_after_keras", action="store_true")
    args = parser.parse_args()

    if args.torch_before_keras:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from keras.losses import MSE
    from keras.optimizers import RMSprop
    from keras import Input, Model
    from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

    if args.torch_after_keras:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Code from previous post here
```

(yes, this code looks horrible, but there is no clean way to experiment with imports).

Running this script with `--torch_after_keras` flag imports Torch and sets flags after the Keras import. This flag
has no effect on reproducibility.

`--torch_before_keras` gives much more interesting and promising results. It imports Torch before Keras, and running
script with this argument produces

```text
I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR

======================================================================
ERROR: test_reproducility (__main__.MyTestCase)
----------------------------------------------------------------------
...
tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
         [[{{node conv2d_1/convolution}}]]
----------------------------------------------------------------------
```

It seems that Keras and Torch indeed can share CuDNN session, and initializing this session with Torch first breaks
Keras initialization (at least in these versions, Keras 2.2.4 and Torch 1.1.0).

So, it seems that right now, if you want consistent reproducible results, you'd better use Pytorch.
