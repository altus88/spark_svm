__author__ = 'gena'
import numpy as np
from localSVM import *
from scipy.optimize import check_grad


def gradientCheck():

    def cost(w):
        return func(w, A, u, z, rho)

    def face_grad(w):
        return func_grad(w, A, u, z, rho)

    def numGradient(J, w):

        e = 0.0001
        p = np.zeros_like(w)
        grad_ = np.zeros_like(w)
        for i in range(np.size(w)):
            p[i] = e
            grad_[i] = np.divide(cost(w+p) - cost(w-p), 2*e)
            p[i] = 0
        return grad_

    nSamples = 10
    nFeatures = 2
    nClasses = 2

    A = np.random.rand(nSamples, nFeatures+1)
    u = np.random.rand(nFeatures+1)
    z = np.random.rand(nFeatures+1)
    rho = 1
    w = np.random.rand((nFeatures+1))


    #grad = face_grad(w)
    #nmGrad = numGradient(cost(w),w)
    return check_grad(func, func_grad, w, A, u, z, rho)
    #return np.linalg.norm(nmGrad-grad)/np.linalg.norm(nmGrad+grad);

print gradientCheck()
