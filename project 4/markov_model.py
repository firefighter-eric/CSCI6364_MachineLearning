import numpy as np


def forward(t, m, order, pi):
    alpha = np.empty((len(order), t.shape[0]))
    alpha[0, :] = pi * (m[:, order[0]])
    for i in range(len(order) - 1):
        alpha[i + 1, :] = np.dot(alpha[i, :], t) * m[:, order[i + 1]]
    p = alpha[-1].sum()
    return alpha, p


def backward(t, m, order, pi):
    beta = np.ones((len(order), t.shape[0]))
    t = t.T
    for i in range(len(order) - 2, -1, -1):
        beta[i, :] = np.dot(beta[i + 1, :] * m[:, order[i + 1]], t)
    p = (beta[0, :] * m[:, order[0]] * pi).sum()
    return beta, p


def filtering(t, m, order, pi):
    alpha, p = forward(t, m, order, pi)
    alpha_sum = alpha.sum(axis=1).reshape(-1, 1).dot(np.ones((1, alpha.shape[1])))
    alpha /= alpha_sum
    return alpha


def smoothing(t, m, order, pi):
    alpha, _ = forward(t, m, order, pi)
    beta, _ = backward(t, m, order, pi)
    s = alpha * beta
    s_sum = s.sum(axis=1).reshape((-1, 1)).dot(np.ones((1, alpha.shape[1])))
    s /= s_sum
    return s


def viterbi(t, m, order, pi):
    l_x, l_y = t.shape[0], len(order)
    delta = np.empty((l_y, l_x))
    phi = np.zeros((l_y, l_x), dtype=np.int)
    delta[0, :] = pi * (m[:, order[0]])

    for i in range(l_y - 1):
        tmp = np.dot(delta[i, :].reshape(-1, 1), np.ones((1, l_x)))
        tmp *= t
        phi[i + 1, :] = np.argmax(tmp, axis=0)
        delta[i + 1, :] = np.max(tmp, axis=0) * m[:, order[i + 1]]

    path = np.empty((l_y,), dtype=np.int)
    path[-1] = np.argmax(delta[-1, :])
    for i in range(l_y - 2, -1, -1):
        path[i] = phi[i + 1, path[i + 1]]
    return delta, phi, path
