import numpy as np
from matplotlib import pyplot as plt

from toy_util import *
from toy_model import *
from toy_optimize import *

from tictoc import tic,toc

x0 = np.array([0,10,0])
U = np.random.randn(T,m)*0.3

def display(X, U):
    ts = np.arange(T) * dt
    Ls = [L(X[i], U[i], dt*i) for i in range(T)]

    plt.plot(ts, X[:,0], label='x')
    plt.plot(ts, X[:,1], label='y')
    plt.plot(ts, X[:,2], label='theta')
    plt.plot(ts, U[:,0], label='u')
    plt.plot(ts, Ls, label='L')

    plt.legend()
    plt.show()

X = predict(x0, U)
plt.plot(X[:,0], X[:,1])
plt.show()
try:
    while True:
        X = predict(x0, U)
        U = gogurt(X, U, 4e-4)
        # U = all_at_once(X, U, 1e-4)
        print(C(X,U))

except KeyboardInterrupt:
    X = predict(x0, U)
    plt.plot(X[:,0], X[:,1])
    plt.show()
    # display(X,U)