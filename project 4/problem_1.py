import numpy as np


T = np.array([[0.4, 0.3, 0.3],
              [0.2, 0.6, 0.2],
              [0.1, 0.1, 0.8]])
Order = np.array([[1, 0, 0]])

for i in range(100000):
    p = np.dot(p, T)

