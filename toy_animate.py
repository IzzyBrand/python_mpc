import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from toy_model import *
from toy_util import *
from toy_optimize import *

x0 = np.random.randn(n)*5
U = np.random.randn(T,m)*0
X = predict(x0, U)

def update(i):
    global X,U,x0

    if np.linalg.norm(X[-1,:2]) < 0.5:
        x0 = np.random.randn(n)*5
        U = np.random.randn(T,m)*0
        X = predict(x0, U)

    U = gogurt(X, U, 5e-3)
    # U = all_at_once(X, U, 1e-3)
    X = predict(x0, U)
    plt.clf()

    # dynamically rescale the plot size
    # plt.xlim(np.min(X[:,0])-1, np.max(X[:,0])+1)
    # plt.ylim(np.min(X[:,1])-1, np.max(X[:,1])+1)

    plt.xlim(-20, 20)
    plt.ylim(-20,20)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x0[0], x0[1])
    plt.scatter(0,0)
    plt.plot(X[:,0], X[:,1])


fig = plt.figure()
ax = fig.add_subplot(111)
ani = FuncAnimation(fig, update, frames=1000, interval=50)
plt.show()