from markov_model import *


# test()


# a = pi.dot(M)

def test_1():
    T = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    M = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    Order = [0, 1, 0]
    Pi = np.array([0.2, 0.4, 0.4]).reshape(1, -1)

    print('test_1')
    print('forward')
    alpha, p = forward(T, M, Order, Pi)
    print(alpha)
    print(p)
    print('viterbi')
    delta, phi, path = viterbi(T, M, Order, Pi)
    print(delta)
    print(phi)
    print(path)


def test_2():
    T = np.array([[0, 0.5, 0.5],
                  [0, 0.9, 0.1],
                  [0, 0.1, 0.9]])
    M = np.array([[0.5, 0.5],
                  [0.9, 0.1],
                  [0.1, 0.9]])
    Order = np.array([2, 3, 3, 2, 2, 2, 3, 2, 3])
    Pi = np.array([1, 0, 0]).reshape(1, -1)

    Order -= 2
    alpha, _ = forward(T, M, Order, Pi)
    f = filtering(T, M, Order, Pi)
    s = smoothing(T, M, Order, Pi)
    print(alpha)
    print('test_2')
    print(f)
    print(s)


def test_3():
    # Whack the mole
    T = np.array([[0.1, 0.4, 0.5],
                  [0.4, 0, 0.6],
                  [0, 0.6, 0.4]])
    M = np.array([[0.6, 0.2, 0.2],
                  [0.2, 0.6, 0.2],
                  [0.2, 0.2, 0.6]])
    Order = np.array([1, 3, 3])
    Pi = np.array([1, 0, 0]).reshape(1, -1)

    Order -= 1
    print('test_3')
    print('forward')
    alpha, p_a = forward(T, M, Order, Pi)
    print(alpha, p_a)
    print('backward')
    beta, p_b = backward(T, M, Order, Pi)
    print(beta, p_b)
    print('filtering')
    f = filtering(T, M, Order, Pi)
    print(f)
    print('smoothing')
    s = smoothing(T, M, Order, Pi)
    print(s)


test_1()
# test_2()
# test_3()
