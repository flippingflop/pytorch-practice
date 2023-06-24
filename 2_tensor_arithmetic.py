import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

### Three most common errors of Pytorch
# 1. Tensors not right datatype - tensor.dtype
# 2. Tensors not right shape - tensor.shape
# 3. Tensors not right devide - tensor.device

### Getting information from tensors (Tensor attributes)
# create a tensor
# some_tensor = torch.rand(3, 4)
# print(some_tensor)

# details about tensor
# print(f"Datatype of tensor: {some_tensor.dtype}")
# print(f"Shape of tensor: {some_tensor.size}")
# print(f"Device tensor is on: {some_tensor.device}") # cpu

### Manipulating Tensors (tensor operations)
# Addition
# Subtraction
# Multiplication (element-wise)
# Division
# Matrix Multiplication

# addition
tensor = torch.tensor([1, 2, 3])
print(tensor + 100)

# multiply
print(tensor * 10)

# subtract
print(tensor - 10)

# PyTorch in-built functions
print(torch.mul(tensor, 10))

















