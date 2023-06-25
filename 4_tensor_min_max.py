import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('---- ---- ------ ---- ----')
print('          result')
print('---- ---- ------ ---- ----')

x = torch.arange(0, 100, 10)
print(x)

# get min
print(x.min())
print(torch.min(x))

# get max
print(x.max())
print(torch.max(x))

# get mean
# mean() doesn't work for Long.
# should be converted to float explicitly.
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())

# get sum
print(torch.sum(x))

### Positional min max
print(x.argmin()) # index of minimun value

x2 = torch.arange(1, 100, 10)
print(x.argmin())
print(x[0])









