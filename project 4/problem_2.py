import numpy as np
from markov_model import *

T = np.array([[0.7, 0.3],
              [0.4, 0.6]])
M = np.array([[0.1, 0.4, 0.5],
              [0.7, 0.2, 0.1]])
Order = [0, 2, 2]
Pi = np.array([0.6, 0.4]).reshape(1, -1)

print('test_1')
print('filtering')
alpha, p = forward(T, M, Order, Pi)
print(alpha)
print(p)
print('viterbi')
delta, phi, path = viterbi(T, M, Order, Pi)
print(delta)
print(phi)
print(path)
