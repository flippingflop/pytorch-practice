import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

tensor = torch.tensor([1, 2, 3])

### Matrix multiplication
# two ways of multiplication in neural networks
# 1. Element-wise multiplication
# 2. Matrix multiplication

# # Element-wise multiplication
# print(tensor, "*", tensor)
# print(f"Equals: {tensor * tensor}")

# # Matrix multiplication
# print(torch.matmul(tensor, tensor))
# # by hand
# print((1 * 1) + (2 + 2) + (3 + 3))

# ## case 1 and case 2 results the same, but case 2 is faster x 1000000
# # case 1
# value = 0
# for i in range(len(tensor)):
#     value += tensor[i] * tensor[i]
# print(value)

# # case 2
# print(torch.matmul(tensor, tensor))


# There are two main rules that performing matrix multiplication needs to satisfy:
# 1. the inner dimensions must match.
# (3, 2) @ (3, 2) won't work
# (2, 3) @ (3, 2) will work
# (3, 2) @ (2, 3) will work
# print(torch.Size(tensor))
# print(torch.matmul(torch.rand(3, 2), torch.rand(2, 2)))

# 2. The resulting matrix has the shape of the outer dimensions:
# (2, 3) @ (3, 2) -> (2, 2)
# (3, 2) @ (2, 3) -> (3, 3)
# print(torch.matmul(torch.rand(10, 10), torch.rand(10, 10)))


### Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])
# print(torch.mm(tensor_A, tensor_B)) # cannot be multiplied (mm = matmul)
# print(torch.Size([3, 2]), torch.Size(3, 2))

# To fix tensor shape issue, we can manipulate the shape of one of our tensors using transpose.
# transpose switches the axis or dimensions of a given tensor.
print(tensor_B)
print(tensor_B.T)
# print(torch.mm(tensor_A, tensor_B)) # doesn't work
print(torch.mm(tensor_A, tensor_B.T)) # it works !

# The matrix multiplication operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape}, 'same shape as above', tensor_B.T = {tensor_B.T}")
print(f"Multiplying: {tensor_A.shape} @ {tensor_B.T.shape} <- inner dimensions must match")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")






