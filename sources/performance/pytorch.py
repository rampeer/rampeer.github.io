import unittest
from time import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class Net(nn.Module):

    def __init__(self, in_shape: Tuple[int, int, int], layers=[(5, 3), (5, 3), (5, 3), (5, 3)]):
        super(Net, self).__init__()
        self.convs = nn.ModuleList()
        channels = in_shape[0]
        current_size = in_shape[1] * in_shape[2]
        for filters, kernel in layers:
            self.convs.append(
                nn.Conv2d(channels, filters, kernel))
            current_size = int(current_size / 4)
            channels = filters
        self.fc1 = nn.Linear(current_size * channels, 1)

    def forward(self, x):
        for conv in self.convs:
            old_size = x.shape
            x = F.relu(conv(x))
            x = nn.functional.pad(x, (0, old_size[2] - x.shape[2], 0, old_size[3] - x.shape[3]))
            x = F.max_pool2d(x, 2)
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        return x


class MyTestCase(unittest.TestCase):
    def test_reproducility(self):
        N = 400
        Ys = np.random.normal(size=N)
        Xs = np.random.normal(size=(N, 3, 256, 256)) + Ys[:, np.newaxis, np.newaxis, np.newaxis] / 10.0
        Xs_batches = np.array_split(Xs, N / 16)
        ys_batches = np.array_split(Ys, N / 16)
        model = Net((3, 256, 256)).cuda()

        log("pytorch", "Torch version:" + torch.__version__ + "\n")
        if torch.cuda.is_available():
            log("pytorch", "CUDA available; devices: " + str(torch.cuda.device_count()) + "\n" +
                torch.cuda.get_device_name(0) + "\n")

        with gauge("pytorch", "train"):
            for epoch in range(30):
                optimizer = optim.SGD(model.parameters(), lr=1e-2)
                for X, Y in zip(Xs_batches, ys_batches):
                    optimizer.zero_grad()

                    output = model(torch.tensor(X, dtype=torch.float).cuda())
                    loss = nn.MSELoss()(output, torch.tensor(Y, dtype=torch.float).cuda())

                    loss.backward()
                    optimizer.step()

        with gauge("pytorch", "inference"):
            for _ in range(30):
                for X in Xs_batches:
                    model(torch.tensor(X, dtype=torch.float).cuda()).data.cpu().detach().numpy()


if __name__ == '__main__':
    unittest.main()
