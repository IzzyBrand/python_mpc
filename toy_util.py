import numpy as np
from toy_model import *

########################### next state and gradients ##########################
# [n] get the next state given current state and control
def step(x,u,t):
    return x + dt * f(x,u,t)

# [n x n] the gradient of the next state wrt change in the previous
def dxdx(x,u,t):
    return np.eye(n) + dt * dfdx(x,u,t)

# [n x m] the change in the next state wrt control
def dxdu(x,u,t):
    return dfdu(x,u,t)

#################################### utils ###################################
# get the cost of an entire trajectory
def C(X, U):
    return sum(L(X[i], U[i], dt*i) for i in range(T))

# estimate the trajectory of the vehicle given a starting condition, x0,
# and a sequence of controls U
def predict(x0, U):
    X = np.zeros([T, n])
    X[0,:] = x0
    for i in range(T-1): X[i+1, :] = X[i] + f(X[i], U[i], dt*i)
    return X

#################################### chains ###################################
# [m] given a sequence of states and controls
# calculate the gradient wrt the ith control input
def dCdu(X, U, i):
    # find the changes in cost resulting from change in control
    dC = dLdu(X[i], U[i], dt*i)

    # and then for each subsequent state, j, find the change in cost
    # as a result from change in control at state i

    # we use an acculumlator the avoid recalculating redundant terms
    # in the chain rule (dLdx * dxdx ... dxdx * dxdu)
    chain_accumulator = dxdu(X[i], U[i], dt*i)
    for j in range(i+1, T):
        dC += dLdx(X[j], U[j], dt*j) @ chain_accumulator
        chain_accumulator = dxdx(X[j], U[j], dt*j) @ chain_accumulator

    return dC

