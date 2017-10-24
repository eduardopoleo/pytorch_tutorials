from __future__ import print_function
import torch

# x = torch.Tensor(5,3)
# print(x)

x = torch.rand(5,3)
print(x)
print(x.size())

y = torch.rand(5,3)
print(x + y)

print(torch.add(x,y))

result = torch.Tensor(5,3)

print('Addition result')
print(result)

result1 = torch.add(x,y, out=result)

print('result1')
print(result1)

print('Numpy')
print(x[:, 1])


# Converting torch Tensor to numpy array
a = torch.ones(5)
print(a)

print('Numpy array from torch tensor')
b = a.numpy()
print(b)

print('Addition in place')
a.add_(1) # mutates the caller.
print(a)
print(b)

print('See how tensor change from changing np arrays')
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)

np.add(a, 1, out=a)

print(a)
print(b)

print('Cuda stuff')

# if torch.cuda.is_available():
#     x = x.cuda()
#     y = y.cuda()
#     z = x + y
#
# print(z)
