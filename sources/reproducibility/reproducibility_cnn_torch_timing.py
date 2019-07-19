import random
import unittest
from time import time

import numpy as np

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


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
    def time_training(self, do_fix_seeds):
        if do_fix_seeds:
            fix_seeds(42)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
        Ws = np.random.normal(size=(20*20*3, 1))
        net = Net(in_shape=20).cuda()
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        t = time()

        for _ in range(1000):
            optimizer.zero_grad()
            Xs = np.random.normal(size=(10, 3, 20, 20))
            Ys = np.dot(Xs.reshape((10, 20 * 20 * 3)), Ws) + np.random.normal(size=(10, 1))

            output = net(torch.tensor(Xs, dtype=torch.float).cuda())
            loss = nn.MSELoss()(output, torch.tensor(Ys, dtype=torch.float).cuda())

            loss.backward()
            optimizer.step()

        return time() - t


if __name__ == '__main__':
    t = MyTestCase()
    non_determenistic = [t.time_training(False) for _ in range(100)]
    print(f"Non-deterministic training times: {np.mean(non_determenistic)} +- {np.sqrt(np.var(non_determenistic))}")

    determenistic = [t.time_training(True) for _ in range(100)]
    print(f"Deterministic training times: {np.mean(determenistic)} +- {np.sqrt(np.var(determenistic))}")
