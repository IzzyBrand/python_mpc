import numpy as np
import sympy as sp
from model_base import ModelBase, SympyModelBase

# a helper function that takes an angle and bounds it [-pi, pi]
def wrap(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def sym_wrap(theta):
    return sp.Mod(theta + sp.pi, 2*sp.pi) - sp.pi

class Toy(ModelBase):
    """
    In this simple toy model, the state is a 3-vector x, y, theta
    and the control input is a 1-vector steering angle which changes theta.
    The objective function is to drive x,y,theta to zero at the last time-step, 
    while minimizing the steering angle at each timestep.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.n = 3
        self.m = 1

    ########################### dynamics and gradients ###########################
    # [n] dynamics given current state and control
    def f(self, x, u, t):
        theta = x[2]
        return np.array([np.cos(theta + u[0]), np.sin(theta + u[0]), u[0]])

    # [n x m] change in dynamics wrt control
    def dfdu(self, x, u, t):
        theta = x[2]
        return np.array([-np.sin(theta + u[0]), np.cos(theta + u[0]), 1])

    # [n x n] change in dynamics wrt state
    def dfdx(self, x, u, t):
        theta = x[2]
        return np.array([[0, 0, -np.sin(theta + u[0])],
                         [0, 0, np.cos(theta + u[0])],
                         [0, 0, 0]])

    ############################# loss and gradients #############################
    # the loss of the current state and control.
    def L(self, x, u, t):
        control_loss = u[0]**2
        position_loss = (x[0]**2 + x[1]**2) * (t==self.dt*(self.T-1))
        return np.array([position_loss + control_loss])

    # [m] change in loss wrt control
    def dLdu(self, x, u, t):
        return np.array([2*u[0]])

    # [n] change in loss wrt state
    def dLdx(self, x, u, t):
        return np.array([2*x[0], 2*x[1], 0]) * (t==self.dt*(self.T-1))


class SympyToy(SympyModelBase):
    """
    This is the same dynamics as implemented above, but we use sympy to
    to automatically find the gradients for us
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.n = 3
        self.m = 1
        self.setup()

    def sym_f(self, x, u, t):
        theta = x[2]
        return sp.Matrix([sp.cos(theta + u[0]), sp.sin(theta + u[0]), u[0]])

    def sym_L(self, x, u, t):
        control_loss = u[0]**2
        pos_loss = x[0]**2 + x[1]**2

        end_loss = sp.Piecewise((0, t < self.dt*(self.T-1)), (pos_loss, True))
        return sp.Matrix([control_loss + end_loss])
