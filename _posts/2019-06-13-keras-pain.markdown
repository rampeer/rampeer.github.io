---
layout: post
title:  "Practical issues of reproducibility"
date:   2019-06-13 00:00:00 +0300
categories: 
---
We are on the quest towards reproducibility.

As I described in [the previous post]({% post_url 2019-05-29-random-seeds %}), fixing random seeds is not enough because
operation parallelization also throws problems in our way. Because the order of operations is not well-defined,
 we may end up with slightly different results.

To be specific, it's an issue with the backend of our library that translates high-level Keras/Pytorch network
description and training information into lower-level commands, CuDNN.

Overall, the pipeline looks like a layered pie:

```
Keras/Pytorch - top-level library you are usually interacting with
Tensorflow/Aten - back-end for these libraries
CuDNN - extention of CUDA for deep learning
CUDA - platform for parallel computations using GPU
Drivers - hardware-specific piece of software that is needed for universal GPU API
Hardware - your RTX 2080 Ti
```

There are deterministic (but slower) algorithm implementations in CuDNN. Because we do not interact with 
CuDNN directly, we have to tell our library of choice to use specific implementations. In other words, we have to
turn on the "I'm ok with slower training, give me consistent results every run" flag.

Unfortunately, Keras does not have that functionality (yet), as described in 
[these](https://github.com/tensorflow/tensorflow/issues/18096) 
[issues](https://github.com/tensorflow/tensorflow/issues/12871).

Looks like it's time for Pytorch to shine. It has settings that will enable the use of CuDNN deterministic implementations:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Let's write a simple network with a single convolution, and train it on random data. The exact architecture or data do not matter much, as we are just testing reproducibility.

```python
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

def get_weight_sums():
    return np.sum([np.sum(x.data.cpu().detach().numpy()) for x in net.parameters()])
```

We'll use synthetic data to train the network:

```python
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
```

[Full script](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_torch.py)

After initialization we ensure that sum of weights is equal to a specific value. Similarly, after training the 
network we check the model weights. If there is any discrepancies or randomness, the script will throw an exception.

Running `python3 reproducibility_cnn_torch.py` gives "OK" consistently, which means training neural network in PyTorch 
gives exactly the same weights.

Therefore, Torch has built-in reproducibility support, and Keras does not have such functionality right now.

*But wait!* There are [speculations](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/45663) 
that because Keras and Torch share back-end, setting these variables in PyTorch
will affect Keras. It could be the case if the variables affected some process-wide parameters.
This way, we can make use of these flags AND use Keras to write models in a painless fashion. 
First of all, let's check out how these flags work. Time to go deeper! 

Setting deterministic / benchmark flags from Python side
[calls functions defined in the C sources](https://github.com/pytorch/pytorch/blob/3a0b27b73d901ab99b6c452b7e716058311e3372/torch/backends/cudnn/__init__.py#L481). 
Here, the state
[is remembered internally](https://github.com/pytorch/pytorch/blob/3a0b27b73d901ab99b6c452b7e716058311e3372/aten/src/ATen/Context.cpp#L67).
In other words, this flag does not affect global state (environment variables, for example).

After that, this flag affects [choice of algorithm](https://github.com/pytorch/pytorch/blob/85528feb409d2a44e2a35637e0768d6de8d92039/aten/src/ATen/native/cudnn/Conv.cpp#L460)
during convolution.

So, it seems that PyTorch settings should not affect global internals.

To be completely sure, let's check it in practice. Let's take 
[previous Keras script](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn.py):
and modify import section:

```python
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
```

(Yes, this code looks horrible, but there is no clean way to experiment with imports).

[Full script](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_import_test.py):

Running this script with `--torch_after_keras` flag imports Torch and changes CuDNN setting after the Keras import. 
This option has no effect on reproducibility.

`--torch_before_keras` imports Torch before Keras, and running script with this argument produces error:

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

That's quite peculiar. I haven't dug It seems that Keras and Torch indeed can share CuDNN session, 
and initializing the session with Torch first breaks
Keras initialization (at least in these versions, Keras 2.2.4 and Torch 1.1.0).

So, it seems that right now, if you want consistent reproducible results, you'd better use Pytorch.
