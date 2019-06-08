import numpy as np
import sympy as sp
from model_base import SympyModelBase

# a helper function that takes an angle and bounds it [-pi, pi]
def wrap(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def sym_wrap(theta):
    return sp.Mod(theta + sp.pi, 2*sp.pi) - sp.pi

class Rocket(SympyModelBase):

    def __init__(self, *args):
        super().__init__(*args)
        self.n = 6
        self.m = 2

        self.mass = 1
        self.I = 1
        self.l = 1
        self.g = 9.8

        self.setup()


    def sym_f(self, x, u, t):
        _, _, theta, xdot, ydot, thetadot = x
        thrust, gimbal = u

        p_thrust = thrust * sp.Heaviside(thrust)

        xddot     = 1/self.mass * (-sp.sin(theta + gimbal) * p_thrust)
        yddot     = 1/self.mass * (sp.cos(theta + gimbal) * p_thrust) - self.g
        thetaddot = 1/self.I * (sp.sin(gimbal) * p_thrust * self.l)

        return sp.Matrix([xdot, ydot, thetadot, xddot, yddot, thetaddot])

    def sym_L(self, x, u, t):
        return x[0]**2 +\
               x[1]**2 +\
               x[2]**2 +\
               x[3]**2 +\
               x[4]**2 +\
               x[5]**2 +\
               u[0]*sp.Heaviside(u[0]) * 1e-5 +\
               u[1]**2 * 100


