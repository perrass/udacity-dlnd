
import numpy as np


def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / len(points)


def step_gradient(b_current, m_current, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - learning_rate * b_gradient
    new_m = m_current - learning_rate * m_gradient
    return [new_b, new_m]


def gradient_decsent_runner(points, starting_b, starting_m, learning_rate, n_iters):
    b = starting_b
    m = starting_m

    for i in range(n_iters):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    # hyperparameters
    learning_rate = 0.0001
    # y = mx + b
    init_b = 0
    init_m = 0
    n_iters = 1000
    [b, m] = gradient_decsent_runner(
        points, init_m, init_b, learning_rate, n_iters)
    print(b)
    print(m)


if __name__ == '__main__':
    run()
