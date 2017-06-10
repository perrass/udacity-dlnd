import numpy as np


print(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]))
print()
print(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1))
print()

a = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
b = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)

print(a[1])
print(b[1])
print()
print(a[1][1])
print(b[1][1])

shape = (2, 2)
c = np.vstack((np.random.random(shape) - 0.5,
               np.random.random((1, shape[1])) - 0.5))

print()
print(c)
print()
print(np.random.random(shape))
