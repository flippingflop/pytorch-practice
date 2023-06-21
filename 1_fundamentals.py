import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

# ### Scalar
# scalar = torch.tensor(7)
# print(scalar.item())

# ### Vector
# vector = torch.tensor([7, 7])
# print(vector)
# print(vector.ndim)
# print(vector.shape)

# ### MATRIX
# MATRIX = torch.tensor([[7, 8], [9, 10]])
# print(MATRIX)
# print(MATRIX.ndim)
# print(MATRIX[1])
# print(MATRIX.shape)

# ### TENSOR
# TENSOR = torch.tensor([[
#     [1, 2, 3],
#     [3, 6, 9],
#     [2, 4, 5]]])
# print(TENSOR)
# print(TENSOR.ndim)
# print(TENSOR.shape)

# ### Random tensors
# # Create random tensor of specific size
# random_tensor = torch.rand(3, 4) #(3, 4, 5)... etc
# print(random_tensor)
# print(random_tensor.ndim)

# # Random tensor similar to shape of image tensor
# random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channels (R, G, B)
# print(random_image_size_tensor.shape)
# print(random_image_size_tensor.ndim)

# ### Zeros and ones
# # tensor of all zeros
# zero = torch.zeros(size=(3, 4))
# print(zero)

# # tensor of all ones
# ones = torch.ones(size=(3, 4))
# print(ones)
# print(ones.dtype)

# ### Creating range of tensors and tensors-like
# one_to_ten = torch.arange(start = 1,end = 10, step = 1)
# print(one_to_ten)
# ten_zeros = torch.zeros_like(one_to_ten)
# print(ten_zeros)

### Tensor datatypes
# float_32_tensor = torch.tensor(
#     [3.0, 6.0, 9.0],
#     dtype=None, # specify datatype. e.g. float32, float16 etc
#     device=None, #
#     requires_grad=False)
# print(float_32_tensor.dtype)

# int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
# print(int_32_tensor)









