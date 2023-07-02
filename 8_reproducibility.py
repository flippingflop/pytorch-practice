import torch
import numpy as np

## Reproducibility (trying to take random out of random)

## How neural network learns:
# 1. start with random numbers
# 2. tensor operations
# 3. update random numbers and make them represents better
# repeat

print('\n\n---- ---- div ---- ----\n\n')
print(torch.rand(3, 3))

# Two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B) # False

### How to make random reproducible tensors
RANDOM_SEED = 77 # Set random seed. can be any value

# generate random tensor with fixed seed
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

# the seed has to be set again.
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

# and not it prints True
print(random_tensor_C == random_tensor_D)










