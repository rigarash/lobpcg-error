import numpy as np
from scipy.sparse.linalg import lobpcg
from scipy.linalg import toeplitz


A = toeplitz([10.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# This fails
X = np.array([[1.0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0, 1.0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
X = X.T
# This works(for high probability)
#X = np.random.rand(20, 2)
val, vec = lobpcg(A, X, largest=False, verbosityLevel=11, tol=1.0e-4)
