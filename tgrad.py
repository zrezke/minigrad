from tinygrad.tensor import Tensor
import numpy as np

np.random.seed(1337)
inp = Tensor(np.array([[1, 1, 1]]).T, requires_grad=True)
print(inp.data)
# weights
l1 = Tensor(np.random.rand(2, 3), requires_grad=True)
print(f"L1: {l1.data}")
bias_l1 = Tensor(np.zeros((2, 1)), requires_grad=True)
# print(f"BIAS L1: {bias_l1}")
l2 = Tensor(np.random.rand(1, 2), requires_grad=True)
# bias_l2 = Tensor(np.zeros((8, 1)))
y = Tensor(np.array([[1], [0]]), requires_grad=True)

d1 = l1.matmul(inp)#.sum()#.tanh()  # .dot(l2).add(bias_l2).tanh()
print(f"1st dot: {d1.data}")
d2 = l2.matmul(d1)#.sum()
out = d2.sum()
print(f"out: {out.data}")
out.backward()
print(l1.grad.data)
print(f"l2 grad: {l2.grad.data}")
print(f"l1 grad: {l1.grad.data}")
print(f"d1.grad: {d1.grad.data}")
print(f"d2.grad: {d2.grad.data}")
