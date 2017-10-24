import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)

print(x)
# 1 1
# 1 1

y = x + 2
print(y)

# 3 3
# 3 3

#  y = x + 2
#  z = 3 * (x + 2)^ 2

z = y * y * 3 # this looks like the . multiplication

out = z.mean()

print('z and out')
print(z, out)

# 27 27
# 27 27

# 27

# x => y = x + 2 => z = y * y * 3 => out = mean(z) # all expressions depend on x
# set x and the rest falls in place.

# out = sum(z) / 4
# z = 3 (x + 2)^2
# dout/dx = 3/2 (x + 2) evaluated at 1 = 4.5

# TODO: Understand this better
# This calculates the backwards propagation through the NN layers
out.backward(torch.Tensor([2.0])) # point at which the bawards is evaluated

print('Printing x grad')

# this calculates the gradient. once the backwards propagation is in place
print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
# y.data.norm() => sqrt(a^2 + b^2 + c^2)
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
