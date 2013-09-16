__author__ = 'gena'
from cvxopt import matrix, spdiag, mul, div, log, blas, lapack, solvers, base
import numpy as np

def pos(X):
    '''
    Return a positive part
    Parameters
    ----------
    X: array-like
    '''
    return np.maximum(X, np.zeros_like(X))

def face_func(x_var, A, u, z, rho):
    return np.sum(pos(np.dot(A, np.atleast_2d(x_var).T) + 1)) + (rho / float(2)) * np.sum(np.power(x_var - z + u, 2))


def func_grad(x_var, A, u, z, rho):
    res = (np.dot(A, np.atleast_2d(x_var).T) + 1)
    ind = np.where(res >= 0)[0]
    return np.sum(A[ind, :], 0) + rho * (x_var - z + u)

nSamples = 1000
nFeatures = 10


A = np.random.rand(nSamples, nFeatures+1)
u = np.random.rand(nFeatures+1)
z = np.random.rand(nFeatures+1)
rho = 1
w = np.random.rand((nFeatures+1))


print func_grad(w, A, u, z, rho)

