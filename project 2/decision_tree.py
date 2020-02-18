import numpy as np


def h(p):
    if p == 0:
        return 0
    else:
        return - p * np.log2(p)


def entropy(arr):
    arr = np.array(arr)
    p = arr / sum(arr)
    ent = []
    for n in p:
        ent.append(h(n))
    return sum(ent)


a = np.array([49, 51])
H_a = entropy(a)
a1 = np.array([24, 1])
H_a1 = entropy(a1)
a2 = np.array([25, 50])
H_a2 = entropy(a2)
H_YX = 0.25 * H_a1 + 0.75 * H_a2
G_a = H_a - H_YX

nums = np.array([3, 4, 4, 1, 0, 1, 3, 5])
nums_X1 = np.array([7, 5, 1, 8])
nums_X2 = np.array([7, 5, 3, 6])
H_Y = entropy([12, 9])
H_X1 = 8 / 21 * entropy([7, 1]) + 13 / 21 * entropy([5, 8])
H_X2 = 10 / 21 * entropy([7, 3]) + 11 / 21 * entropy([5, 6])

# H_X2 = entropy(nums_X2)
IG_X1 = H_Y - H_X1
IG_X2 = H_Y - H_X2
