import numpy as np


# improves the cost of a trajectory
class OptimizerBase:

    def __init__(self, model):
        self.model = model

    # [T x m] return an improved control sequence
    def step(self, x0, U, step_size):
        return np.zeros_like(U)


# takes a gradient descent step on each control in sequential order
class Gogurt(OptimizerBase):

    def step(self, x0, U, step_size):
        X = self.model.predict(x0, U)
        for i in range(self.model.T):
            U[i] -= step_size * dCdu(X, U, i)
            X = predict(X[0], U)

        return U


# takes a gradient descent step on all controls simultaneously
class AllAtOnce(OptimizerBase):

    def step(self, x0, U, step_size):
        dCdU = [self.model.dCdu(X, U, i) for i in range(self.model.T)]
        return U - step_size * np.array(dCdU)
