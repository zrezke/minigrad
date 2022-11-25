import numpy as np


class Function:

    @classmethod
    def backward(cls, *args):
        raise NotImplementedError(
            f"backward isn't implemented in {cls.__name__}")

    @classmethod
    def apply(cls) -> np.ndarray:
        raise NotImplementedError(
            f"apply isn't implemented in {cls.__name__}")


class MSE(Function):

    @classmethod
    def backward(cls, op1: 'Tensor', op2: 'Tensor', previous_grad: np.ndarray):
        op1.grad = np.divide(np.multiply(np.sum(np.subtract(op2.data, op1.data), 0), -2), op1.data.shape[0])
        # print(f"MSE OP1: {op1.grad}")
        op1.backward()

    @classmethod
    def apply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(np.power(np.subtract(b, a), 2), 0) / a.shape[0]


class Tanh(Function):

    @classmethod
    def backward(cls, op: 'Tensor', previous_grad: np.ndarray):
        op.grad = 1 - np.tanh(op.data) * np.tanh(op.data)
        # print(f"TANH OP: {op.grad}")
        op.backward()

    @classmethod
    def apply(cls, op: 'Tensor'):
        return np.tanh(op.data)


class Dot(Function):

    @classmethod
    def backward(cls, op1: 'Tensor', op2: 'Tensor', previous_grad: np.ndarray):
        # op1 is the activations(L-1), op2 is the weights 
        op2.grad = op1.data.reshape(1, op1.data.shape[0]).repeat(op2.data.shape[0], 0) * previous_grad
        op1.grad = (op2.data * previous_grad).sum(axis=0)
        op2.grad = np.dot(op1.data, previous_grad.T).T
        op1.backward()
        op2.backward()


    @classmethod
    def apply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)


class Add(Function):

    @classmethod
    def backward(cls, op1: 'Tensor', op2: 'Tensor', previous_grad: np.ndarray):
        op1.grad = previous_grad
        # print(f"ADD OP1: {op1.grad}")
        op1.backward()

        op2.grad = previous_grad
        # print(f"ADD OP2: {op1.grad}")
        op2.backward()

    @classmethod
    def apply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.add(a, b)


class Tensor:

    def __init__(self, data: np.ndarray = None, function: 'Function' = None, function_args: list = None):
        self.data = data
        self.function = function
        self.function_args = function_args
        self.grad = np.ones(data.shape)

    def __str__(self):
        return str(self.data)

    def backward(self):
        if self.function:
            self.function.backward(*self.function_args, self.grad)

    def dot(self, b: 'Tensor'):
        return Tensor(Dot.apply(b.data, self.data), Dot, [self, b])

    def add(self, b: 'Tensor'):
        return Tensor(Add.apply(self.data, b.data), Add, [self, b])

    def tanh(self):
        return Tensor(Tanh.apply(self.data), Tanh, [self])

    def mse(self, b: 'Tensor'):
        return Tensor(MSE.apply(self.data, b.data), MSE, [self, b])


class SimpleOptimizer:

    def __init__(self, step: float = 0.01):
        self.step = step

    def optimize(self, tensor: Tensor, l1: Tensor):
        tensors = [tensor]
        while tensors:
            tensors[0].grad = np.subtract(tensors[0].data, np.multiply(
                tensors[0].grad, self.step))
            if tensors[0].function_args:
                tensors += tensors[0].function_args
            tensors.pop(0)


# np.random.seed(1337)
# inp = Tensor(np.array([[1, 1, 1]]).T)
# # weights
# l1 = Tensor(np.random.rand(2, 3))
# print(f"L1: {l1}")
# bias_l1 = Tensor(np.zeros((2, 1)))
# # print(f"BIAS L1: {bias_l1}")
# # l2 = Tensor(np.random.rand(8, 6))
# # bias_l2 = Tensor(np.zeros((8, 1)))
# y = Tensor(np.array([[1],[0]]))

# out = inp.dot(l1).add(bias_l1).tanh()  # .dot(l2).add(bias_l2).tanh()
# print(out)
# err = out.mse(y)
# print(err)
# err.backward()

# print("GRADS:")
# print(f"MSE: {err.grad}")
# print(f"Tanh {out.grad}")
# print(f"ADD RESULT: {out.function_args[0].grad}")
# print(f"ADD OP1 (DOT RES): {out.function_args[0].function_args[0].grad}")
# print(f"ADD OP2 (BIAS): {out.function_args[0].function_args[1].grad}")
# print(
#     f"DOT WEIGHTS: {out.function_args[0].function_args[0].function_args[1].grad}")
# print(
#     f"DOT ACTIVATIONS: {out.function_args[0].function_args[0].function_args[0].grad}")
