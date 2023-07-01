import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

# Create tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print(x.shape)

# index the tensor
print(x[0])
print(x[0][0])
print(x[0][0][0]) # most inner bracket

# select ALL of specific dimension
print(x[:, :, 1])









