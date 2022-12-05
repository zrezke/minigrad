# minigrad
Minigrad is a small autograd engine I implemented in python + numpy as a learning exercise in my latest quest to learn about AI (AI is getting to good to not know how it works)

## Current status
- It works for a very simple neural net that learns the xor function -> see [xor_test.py](xor_test.py)
- Dot product backward only works for nx1 dimensional "tensors"
- Probably some other backwards are broken in a similar way
