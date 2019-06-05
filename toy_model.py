import numpy as np
from model_base import ModelBase

# a helper function that takes an angle and bounds it [-pi, pi]
def wrap(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

class Toy(ModelBase):

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
        angle_loss = 10*wrap(x[2])**2 * (t==self.dt*(self.T-1))
        return np.array([angle_loss + position_loss + control_loss])

    # [m] change in loss wrt control
    def dLdu(self, x, u, t):
        return np.array([2*u[0]])

    # [n] change in loss wrt state
    def dLdx(self, x, u, t):
        return np.array([2*x[0], 2*x[1], 20*wrap(x[2])]) * (t==self.dt*(self.T-1))
