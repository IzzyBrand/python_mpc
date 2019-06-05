import numpy as np

from toy_util import *
from toy_model import *

def dCdU(X, U):
    return np.array([dCdu(X, U, i) for i in range(T)])

# update all the controls at once by calculating the change
# in the loss wrt each control separately
def all_at_once(X, U, step_size):
    return U - step_size * dCdU(X, U)

# update each control from the first to the last
def gogurt(X, U, step_size):
    for i in range(T):
        U[i] -= step_size * dCdu(X, U, i)
        X = predict(X[0], U)

    return U

