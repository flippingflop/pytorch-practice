import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

# Reshape - reshapes the original tensor as required
# View - returns a new shape of tensor but the original tensor is remained same in the memory
# Stacking - combine multiple tensor horizontally or vertically.
# Squeeze - remove all 1 dimensions from a tensor
# Unsqueeze - add 1 dimension to a tensor
# Permute - return a view of the input with dimensions permuted (swapped) in a certain way.

x = torch.arange(1., 10.)
print(x)
print(x.shape)

# Add an extra dimension
x_reshaped = x.reshape(1, 9)
print(x_reshaped)

# x_reshaped2 = x.reshape(3, 4)
# print(x_reshaped2)

print('---- ---- div ---- ----')

# Change the view
x = torch.arange(1., 10.)
z = x.view(1, 9)
print(z, z.shape)
# if you change z, it changes x as well. it's because view shares the memory of original input.
z[:, 0] = 5 # changes index 0 of every element to 5
print(z, x)

print('---- ---- div ---- ----')

# Stack tensors on top of each other.
print(x)
x_stacked = torch.stack([x, x, x, x], dim = 1)
print(x_stacked)

# torch.squeeze: returns a tensor which all the dimension size is 1 removed.
print(f"previous tensor: {x_reshaped}")
print(f"previous shape: {x_reshaped.shape}")

x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# torch.unsqueeze(): add single dimension to a target tensor at a specific dim.
print(f"Prev target: {x_squeezed}")
print(f"prev shape: {x_squeezed.shape}")

x_unsqueezed = x_squeezed.unsqueeze(dim = 0)
print(f"\nnew tensor: {x_unsqueezed}")
print(f"new shape: {x_unsqueezed.shape}")

# torch.permute: rearranges the dimensions of a tensor in a specific order.
x_original = torch.rand(size = (224, 224, 3)) # height, width, color channels
# print(x_original)
x_permuted = x_original.permute(2, 0, 1) # shifts axis. 0 -> 1, 1 -> 2, 2 -> 0
print(x_original.shape)
print(x_permuted.shape)





