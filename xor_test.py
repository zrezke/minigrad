from tensor import Tensor, SimpleOptimizer
import numpy as np
import random


class NNET:

    def __init__(self):
        # self.l1 = Tensor(np.ones((2, 2)))
        self.l1 = Tensor(np.random.rand(2,2))
        self.bias_l1 = Tensor(np.repeat([0], 2, axis=0).reshape((2, 1)))
        # self.l2 = Tensor(np.array([[1, 2]]))
        self.l2 = Tensor(np.random.rand(1,2))
        self.bias_l2 = Tensor(np.repeat([0], 1, axis=0).reshape(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.l1).add(self.bias_l1).tanh().dot(self.l2).add(self.bias_l2).tanh()

    def label_to_tensor(self, label: int) -> Tensor:
        return Tensor(np.array([[label]]))

    def train(self, data: np.ndarray):
        optimizer = SimpleOptimizer(0.1)
        for d in data:
            x = Tensor(d[:2].reshape(2, 1))
            y = self.label_to_tensor(d[2])
            out = self.forward(x)
            print(out)
            mse = out.mse(y)
            print(f"EXPECTED: {y}, got: {out}, MSE: {mse}")
            # print(f"L1: {self.l1}")
            mse.backward()
            optimizer.optimize(mse, self.l1)
            # print(self.l1)


combinations = [[0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0]]
# same dataset for testing and training cuz it's not important to have separate sets
# when I just want to see if my network can learn (if autograd works) - spoiler alert: it doesn't work
dataset = np.repeat(combinations, 100, 0)
np.random.shuffle(dataset)
# dataset = dataset[0:1]
nnet = NNET()

nnet.train(dataset)
x = Tensor(dataset[78][:2].reshape(2, 1))
y = nnet.label_to_tensor(dataset[78][0])
out = nnet.forward(x)
mse = out.mse(y)
print(x, y, out)
