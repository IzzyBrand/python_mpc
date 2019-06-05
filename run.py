import numpy as np
from matplotlib import pyplot as plt

from toy_model import Toy
from optimizer import *

def current_cost(M, x0, U):
	X = M.predict(x0, U)
	return M.C(X, U)

if __name__ == '__main__':
	# parameters
	dt = 0.1
	T = 50

	# specify the model and optimizer
	model = Toy
	optimizer = Gogurt

	# instantiate the model and optimizer
	M = model(dt, T)
	O = optimizer(M)

	# specify start state and the initial plan
	x0 = np.array([0, 10, 0])
	U = np.zeros([T, M.m])

	assert x0.size == M.n

	try:
	    while True:
	   		# improve the plan
	        U = O.step(x0, U, 1e-3)

	        # and print the cost
	        X = M.predict(x0, U)
	        print(M.C(X, U))

	except KeyboardInterrupt:
	    X = M.predict(x0, U)
	    plt.plot(X[:,0], X[:,1])
	    plt.show()
