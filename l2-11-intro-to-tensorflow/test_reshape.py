import numpy as np


w = np.array([[1, 2], [3, 4]])
print(w.reshape(-1, 1))

print(w.reshape(-1, -1))
