import torch
import numpy as np

# Numpy is a numerical computing library.
# data in numpy, want in pytorch tensor -> torch.from_numpy(ndarray)
# pytorch tensor -> numpy -> torch.Tensor.numpy()

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array.dtype) # float64: numpy's default data type.
tensor = torch.from_numpy(array)
print(array, tensor)


print('\n\n---- ---- div ---- ----\n\n')
# change the value of array.
print(array)
print(array + 1)


print('\n\n---- ---- div ---- ----\n\n')
# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor)
print(tensor.dtype) # torch.float32
print(numpy_tensor)
print(numpy_tensor.dtype) # float32


print('\n\n---- ---- div ---- ----\n\n')
print(tensor)
tensor = tensor + 1
print(tensor)




