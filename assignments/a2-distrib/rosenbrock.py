# rosenbrock.py

import argparse
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import numpy as np


def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='rosenbrock.py')
    parser.add_argument('--method', type=str, default='SGD', help='optimizer to use (SGD or ADAM)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args


class Rosenbrock(nn.Module):
    """
    Defines the function sum_{i=1}^{n-1} [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ], which is minimized at x = all ones
    """
    def __init__(self, dim=5):
        super(Rosenbrock, self).__init__()
        self.weight = Parameter(torch.Tensor(dim))
        nn.init.zeros_(self.weight)

    def forward(self, x):
        ones_vec = torch.from_numpy(np.ones([self.weight.shape[0] - 1])).float()
        one_minus_x = ones_vec - self.weight[:-1]
        inner_term = self.weight[1:] - (self.weight[:-1] * self.weight[:-1])
        return 100 * inner_term.dot(inner_term) + one_minus_x.dot(one_minus_x)


if __name__ == '__main__':
    args = _parse_args()

    # Both functions are optimized at [1, 1, 1, 1, 1]. Default dimension for each is 5.
    network = Rosenbrock()
    if args.method == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for iter in range(0, args.epochs):
        value = network.forward(None)
        print("=====Iteration %i=====" % iter)
        print("Current parameters: " + repr(network.weight.data))
        print("Function value: " + repr(value.data))
        network.zero_grad()
        value.backward()
        optimizer.step()
