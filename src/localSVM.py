'''
Created on May 22, 2013

@author: gena
'''
import numpy as np
from scipy.optimize import fmin_l_bfgs_b,fmin_bfgs
import pylab as plb
from scikits.learn import datasets
import time
import random

def plotDesisionLine(data, w, b):
    x1 = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plb.plot(x1, x2)


def plot2D(data, theta=[0, 0, 0] ):
    '''
        theta: array-like, theta=(w,b)
    '''
    X = data[:, 0:2]
    y = data[:, 2]

    pos = np.where(y == 1)
    neg = np.where(y == -1)
    plb.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    plb.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    plb.xlabel('x1')
    plb.ylabel('x2')
    plb.legend(['Positive', 'Negative'])
    # if theta !=0:
    plotDesisionLine(data, theta[0:2], theta[2])
    plb.show()
    #raw_input("Press ENTER to exit") 


#def func(x_var, A, u, z, rho):
    #return np.sum(pos(np.dot(A, np.atleast_2d(x_var).T) + 1)) + (rho / float(2)) * np.sum(np.power(x_var - z + u, 2))

def func(x_var, A, u, z, rho):
    res = np.dot(A, np.atleast_2d(x_var).T) + 1
    ind = np.where(res >= 0)[0]
    return np.sum(pos(res)) + (rho / float(2)) * np.sum(np.power(x_var - z + u, 2)), np.sum(A[ind, :], 0) + rho * (x_var - z + u)


def face_func(x_var, A, u, z, rho):
    return np.sum(pos(np.dot(A, np.atleast_2d(x_var).T) + 1)) + (rho / float(2)) * np.sum(np.power(x_var - z + u, 2))


def func_grad(x_var, A, u, z, rho):
    res = (np.dot(A, np.atleast_2d(x_var).T) + 1)
    ind = np.where(res >= 0)[0]
    return np.sum(A[ind, :], 0) + rho * (x_var - z + u)


def opt_function(x_var, A, u, z, rho):
    pass




def pos(X):
    '''
    Return a positive part
    Parameters
    ----------
    X: array-like
    '''
    return np.maximum(X, np.zeros_like(X))

def pos2(X):
    '''
    Return a positive part
    Parameters
    ----------
    X: array-like
    '''
    return np.maximum(X, np.zeros_like(X))



def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


def transformData(arr):
    '''
      Return a tuple (class,(-class*dataPoints,-class))
      arr: array-like,shape = (nFeatures,)
    '''
    nFeatures = np.size(arr, 1) - 1
    return np.hstack((np.array(-arr[:, -1], ndmin=2).T * arr[:, 0:nFeatures], np.array(-arr[:, -1], ndmin=2).T))


def objective(A, lambda_, x, z):
    return hinge_loss(A, x) + 1 / (2 * lambda_) * np.sum(z ** 2)


def hinge_loss(A, x):
    val = 0;
    for i in range(len(A)):
        val += np.sum(pos(np.dot(A[i], x[:, i]) + 1))
    return val


if __name__ == '__main__':

# data = [parseVector(line) for line in open('data.txt')]
    # data = np.array(data)
    # Parameters
    fileName = "test1"
    over_relaxation = 1
    alpha = 1.2

    isRandomSplit = 0
    warm_start = 1


    loadData = 1
    split = [4, 4]
    percentage = 100
    N_samples = 50000
    plotData = 0
    MAX_ITER = 10000
    nFeatures = 3

    if loadData == 0:
        x, y = datasets.make_classification(n_samples=N_samples, n_features = nFeatures, n_informative=nFeatures, n_redundant=0,
                                        n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y = None,
                                       class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        y[np.where(y == 0)] = -1
        data = np.hstack((x, np.atleast_2d(y).T))
    else:
        with open(fileName) as data_file:
             data = [parseVector(line) for line in data_file]
        data = np.array(data)

    if percentage != 100:
        ind = random.sample(range(N_samples), int(N_samples*percentage/100.))
        plotted_data = data[ind, :]
    else:
        plotted_data = data

    if plotData == 1:
        plot2D(plotted_data)


    N = sum(split)

    if isRandomSplit == 0:
        neg_data_split = np.array_split(data[data[:, -1] == -1], split[0])
        pos_data_split = np.array_split(data[data[:, -1] == 1], split[1])
        A = list()

        A.extend(neg_data_split)
        A.extend(pos_data_split)

    else:
        import random
        random.shuffle(data)
        A = np.array_split(data, N)


    A = map(transformData, A)


    # N = np.size(A)

    u = np.zeros((nFeatures + 1, N))
    z = np.zeros((nFeatures + 1, N))
    lambda_ = 1

    rho = 1

    x_var = np.zeros((nFeatures + 1, 1))#np.random.rand(nFeatures + 1, 1)
    x = np.zeros((nFeatures + 1, N))
    i = 0

    QUIET = 0
    ABSTOL = 1e-4
    RELTOL = 1e-2

    objval = list()
    r_norm = list()
    s_norm = list()
    eps_pri = list()
    eps_dual = list()

    #[ 0.7231721   0.64773677 -3.96192998]



    sumTime = 0
    for j in (range(MAX_ITER)):
        maxTime = 0
        i = 0
        for arr in A:
            startTime = time.time()

            res = fmin_l_bfgs_b(func, x_var, None,\
                                (arr, u[:, i], z[:, 0], rho), \
                                approx_grad=0, bounds=None, factr=1e7, pgtol=1e-5, \
                                epsilon=1e-8, iprint=0, maxiter=10000,\
                                disp=0)


            x[:, i] = res[0]
            i += 1

            elapsedTime = time.time() - startTime
            if elapsedTime > maxTime:
                maxTime = elapsedTime




        # z-update with relaxation
        print maxTime
        sumTime += maxTime
        zold = z

        if over_relaxation == 1:
            x_hat = alpha * x + (1 - alpha) * zold
        else:
            x_hat = x



        z = np.tile(float(N * rho) / (1 / lambda_ + N * rho) * np.mean(x_hat + u, 1), (N, 1)).T
        #  u-update
        u = u + (x_hat - z)
        #print u
        #print "---------------"
        #print z


        objval.append(objective(A, lambda_, x, z[:, 0]))
        r_norm.append(np.linalg.norm(x - z, 2))
        s_norm.append(np.linalg.norm(-rho * (z - zold), 2))
        #print np.linalg.norm(x)
        #print np.linalg.norm(-z)
        #print np.sqrt(nFeatures+1)*ABSTOL + RELTOL*np.max(np.linalg.norm(x),np.linalg.norm(-z))
        #if j<9:
        #eps_pri.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*max(np.linalg.norm(x,2),np.linalg.norm(-z,2)))
        #else:
        eps_pri.append(np.sqrt(nFeatures + 1) * ABSTOL + RELTOL * max(np.linalg.norm(x, 2), np.linalg.norm(-z, 2)))

        eps_dual.append(np.sqrt(nFeatures + 1) * ABSTOL + RELTOL * np.linalg.norm(rho * u, 2))

        print str(j) + " " + str(objval[j]) + " " + str(r_norm[j]) + " " + str(s_norm[j]) \
              + " " + str(eps_pri[j]) + " " + str(eps_dual[j])

        if (r_norm[j] < eps_pri[j] and s_norm[j] < eps_dual[j]):
        # if (r_norm[j] < 0.001 and s_norm[j] < 0.001):
            break;
        xave = np.mean(x, 1)
    print "Elapsed time: " + str(sumTime)

    if plotData == 1:
        plot2D(plotted_data, xave)

    saveData = not loadData
    if saveData:
       np.savetxt(fileName, data)



