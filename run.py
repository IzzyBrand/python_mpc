import numpy as np
from matplotlib import pyplot as plt

from toy_model import Toy, SympyToy
from rocket import Rocket
from optimizer import *

def current_cost(M, x0, U):
    X = M.predict(x0, U)
    return M.C(X, U)

if __name__ == '__main__':
    # parameters
    dt = 0.1
    T = 10

    # specify the model and optimizer
    model = SympyToy
    optimizer = Gogurt

    # instantiate the model and optimizer
    M = model(dt, T)
    O = optimizer(M)

    # specify start state and the initial plan

    # for the rocket
    # x0 = np.array([0, 0, 0.1, 0, 0, 0])
    # U = np.array([9.8*np.ones(T), np.zeros(T)]).T

    # for the toy
    x0 = np.array([5, 5, 0])
    U = np.random.randn(T, 1) * 0.1

    assert x0.size == M.n

    try:
        while True:
            # improve the plan
            U = O.step(x0, U, 1e-2 / T)

            # and print the cost
            X = M.predict(x0, U)
            print(M.C(X, U))

    except KeyboardInterrupt:
        X = M.predict(x0, U)

        plt.xlim(np.min(X[:,0])-1, np.max(X[:,0])+1)
        plt.ylim(np.min(X[:,1])-1, np.max(X[:,1])+1)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.plot(X[:,0], X[:,1])
        plt.show()
