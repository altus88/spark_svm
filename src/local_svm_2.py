__author__ = 'gena'
'''
Created on May 22, 2013

@author: gena
'''
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pylab as plb

def plotDesisionLine(data,w,b):
    x1 = np.linspace(min(data[:,0]), max(data[:,0]), 100)
    x2 = -(w[0]*x1 + b)/w[1]
    plb.plot(x1,x2)

def plot2D(data,theta):
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
    plotDesisionLine(data,theta[0:2],theta[2])
    plb.show()
    #raw_input("Press ENTER to exit")


def func(x_var,A,u,z,rho):
    return np.sum(pos(np.dot(A,np.atleast_2d(x_var).T) + 1)) +  (rho/float(2))*np.sum(np.power(x_var -z+u,2))

def pos(X):
    '''
    Return a positive part
    Parameters
    ----------
    X: array-like
    '''
    return np.maximum(X,np.zeros_like(X))

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def transformData(arr):
    '''
      Return a tuple (class,(-class*dataPoints,-class))
      arr: array-like,shape = (nFeatures,)

    '''
    nFeatures = np.size(arr,1) - 1
    return np.hstack((np.array(-arr[:,-1],ndmin = 2).T*arr[:,0:nFeatures],np.array(-arr[:,-1],ndmin=2).T))

def objective(A, lambda_, x, z):
    return hinge_loss(A,x) + 1/(2*lambda_)*np.sum(z**2)

def hinge_loss(A,x):
    val = 0;
    for i in range(len(A)):
        val+= np.sum(pos(np.dot(A[i],x[:,i]) + 1))
    return val

data =[parseVector(line) for line in open('data.txt')]

data = np.array(data)


#plot2D(data,np.array([ 0.66285737,  0.65203543, -3.86435969]))

split = [3 ,3]
N = sum(split)


neg_data_split = np.array_split(data[data[:,-1]==-1],split[0])
pos_data_split = np.array_split(data[data[:,-1]==1],split[1])

A = list()

A.extend(neg_data_split)
A.extend(pos_data_split)

A = map(transformData,A)

nFeatures = 2
N = np.size(A)

u = np.zeros((nFeatures+1,N))
z = np.zeros((nFeatures+1,N))
lambda_ = 1

rho = 1
alpha = 1.4
x_var = [0,0,0]
x = np.zeros((nFeatures+1,N))
i = 0

QUIET    = 0
MAX_ITER = 1000
ABSTOL   = 1e-4
RELTOL   = 1e-2

objval = list()
r_norm = list()
s_norm = list()
eps_pri = list()
eps_dual = list()

#[ 0.7231721   0.64773677 -3.96192998]
for j in (range(MAX_ITER)):
    i=0
    if (j==30):
        print i
    for arr in A:
        res  = fmin_l_bfgs_b(func, x_var, None, \
                               (arr,u[:,i],z[:,0],rho),\
                               approx_grad=1,bounds=None, m=10, factr=1e7, pgtol=1e-5,\
                               epsilon=1e-6,iprint=0, maxfun=15000, maxiter=10000,\
                               disp=0, callback=None)
        x[:,i] = res[0]
        i+=1


    # z-update with relaxation
    zold = z
    x_hat = alpha*x +(1-alpha)*zold
    z = np.tile(float(N*rho)/(1/lambda_ + N*rho)*np.mean( x_hat + u, 1 ),(N,1)).T
    #  u-update
    u = u + (x_hat - z)
    #print u
    #print "---------------"
    #print z


    objval.append(objective(A, lambda_, x, z[:,0]))
    r_norm.append(np.linalg.norm(x-z,2))
    s_norm.append(np.linalg.norm(-rho*(z-zold),2))
    #print np.linalg.norm(x)
    #print np.linalg.norm(-z)
    #print np.sqrt(nFeatures+1)*ABSTOL + RELTOL*np.max(np.linalg.norm(x),np.linalg.norm(-z))
    #if j<9:
    #eps_pri.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*max(np.linalg.norm(x,2),np.linalg.norm(-z,2)))
    #else:
    eps_pri.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*max(np.linalg.norm(x,2),np.linalg.norm(-z,2)))

    eps_dual.append(np.sqrt(nFeatures+1)*ABSTOL + RELTOL*np.linalg.norm(rho*u,2))


    print str(j) + " " + str(objval[j])+ " " + str(r_norm[j])+ " " + str(s_norm[j])\
    + " " + str(eps_pri[j])+ " " + str(eps_dual[j])

    if (r_norm[j] < eps_pri[j] and s_norm[j] < eps_dual[j]):
        break;
    xave = np.mean(x,1)
plot2D(data,xave)