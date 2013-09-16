__author__ = 'gena'
from sklearn.svm import LinearSVC
import numpy as np
from scikits.learn import datasets
from localSVM import *


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


N_samples = 1000000
x, y = datasets.make_classification(n_samples=N_samples, n_features = 2, n_informative=2, n_redundant=0,
                                    n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y = None,
                                   class_sep=2.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
y[np.where(y == 0)] = -1
data = np.hstack((x, np.atleast_2d(y).T))


#plot2D(data)

svm = LinearSVC()
#data = [parseVector(line) for line in open('test.txt')]
#data = np.array(data)
#x = data[:, :-1]
#y = data[:, -1]
svm.verbose = 1
print "Fitting SVM"
svm.fit(x,y)

#print svm.predict([1, 2])
print "Plotting data"
percentage = 1
ind = random.sample(range(N_samples), int(N_samples*percentage/100.))
data = data[ind, :]
#print svm.raw_coef_
plot2D(data, svm.raw_coef_[0])




