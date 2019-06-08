import numpy as np
import sympy as sp

class ModelBase:

    def __init__(self, dt, T):
        self.n = 1      # dimensionality of state
        self.m = 1      # dimensionality of control
        self.dt = dt    # duration of timestep
        self.T = T      # number of steps to planning horizon

    ############################ TO BE IMPLEMENTED ############################

    # [n] dynamics given current state and control
    def f(self, x, u, t):
        pass

    # [n x m] change in dynamics wrt control
    def dfdu(self, x, u, t):
        pass

    # [n x n] change in dynamics wrt state
    def dfdx(self, x, u, t):
        pass

    # the loss of the current state and control
    def L(self, x, u, t):
        pass

    # [m] change in loss wrt control
    def dLdu(self, x, u, t):
        pass

    # [n] change in loss wrt state
    def dLdx(self, x, u, t):
        pass

    ########################## END TO BE IMPLEMENTED ##########################

    # [n x n] the gradient of the next state wrt change in the previous
    def dxdx(self, x, u, t):
        return np.eye(self.n) + self.dt * self.dfdx(x, u, t)

    # [n x m] the change in the next state wrt control
    def dxdu(self, x, u, t):
        return self.dfdu(x, u, t)

    # [n] get the next state given current state and control
    def step(self, x, u, t):
        return x + self.dt * self.f(x, u, t)

    # [T x n] get the trajectory starting at x0 following control sequence U
    def predict(self, x0, U):
        X = np.zeros([self.T, self.n])
        X[0] = x0
        for i in range(self.T-1): X[i+1] = X[i] + self.f(X[i], U[i], self.dt*i)
        return X

    # calculate the cost of an entire trajectory
    def C(self, X, U):
        return sum(self.L(X[i], U[i], self.dt*i) for i in range(self.T))

    # [m] calculate cost gradient wrt the ith control input
    def dCdu(self, X, U, i):
        # find the changes in cost resulting from change in control
        dC = self.dLdu(X[i], U[i], self.dt*i)


        # and then for each subsequent state, j, find the change in cost
        # as a result from change in control at state i

        # we use an acculumlator the avoid recalculating redundant terms
        # in the chain rule (dLdx * dxdx ... dxdx * dxdu)
        chain_accumulator = self.dxdu(X[i], U[i], self.dt*i)
        for j in range(i+1, self.T):
            dC += self.dLdx(X[j], U[j], self.dt*j) @ chain_accumulator
            chain_accumulator = self.dxdx(X[j], U[j], self.dt*j) @ chain_accumulator

        return dC


# by implementing the dynamics and cost function with sympy operations, we can
# use symbolic differentiation to autonmatically find all the Jacobians
class SympyModelBase(ModelBase):

    def __init__(self, dt, T):
        super().__init__(dt, T)

    ############################ TO BE IMPLEMENTED ############################

    def sym_f(self, x, u, t):
        pass

    def sym_L(self, x, u, t):
        pass

    ########################## END TO BE IMPLEMENTED ##########################

    # takes a sympy symbolic Matrix, and converts it to a function
    # which output a numpy array corresponding to subsituting values in
    # for the symbols
    def convert_sympy_expr_to_np_func(self, mat, x, u, t):
        mat_func = sp.lambdify((x, u, t), mat, modules='sympy')
        return lambda x, u, t: np.array(mat_func(x, u, t), dtype=float).squeeze()

    # calculates the Jacobians symbolically and sets all the relevant functions
    def setup(self):
        x = sp.Matrix(sp.symbols('x0:{}'.format(self.n)))
        u = sp.Matrix(sp.symbols('u0:{}'.format(self.m)))
        t = sp.symbols('t')

        f = self.sym_f(x, u, t)
        dfdx = f.jacobian(x)
        dfdu = f.jacobian(u)

        L = self.sym_L(x, u, t)
        dLdx = L.diff(x)
        dLdu = L.diff(u)

        self.f = self.convert_sympy_expr_to_np_func(f, x, u, t)
        self.L = self.convert_sympy_expr_to_np_func(L, x, u, t)
        self.dfdx = self.convert_sympy_expr_to_np_func(dfdx, x, u, t)
        self.dfdu = self.convert_sympy_expr_to_np_func(dfdu, x, u, t)
        self.dLdx = self.convert_sympy_expr_to_np_func(dLdx, x, u, t)
        self.dLdu = self.convert_sympy_expr_to_np_func(dLdu, x, u, t)