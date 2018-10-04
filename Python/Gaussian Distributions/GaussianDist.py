#Nicholas Klein
#Created 9/17/18, Last edit 9/20/18
#Gaussian Distributions for Machine Learning

import numpy
import matplotlib.pyplot as plt

#Linearly Separable Gaussian Distribution
mean = [-4, 4]
cov = [[1, 3], [3, 1]]
numPoints = 10000
x1, y1 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [3, -3]
cov = [[5, 4], [4, 5]]
numPoints = 10000
x2, y2 = numpy.random.multivariate_normal(mean, cov, numPoints).T

plt.figure()
plt.plot(x1, y1, 'or')
plt.show
plt.plot(x2, y2, 'og')
plt.title('Linearly Separable Gaussian Distribution')
plt.show()

#Non-Linear Separability
mean = [3, 8]
cov = [[3, 4], [2, 1]]
numPoints = 10000
x1, y1 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [0, -1]
cov = [[5, 4], [1, 1]]
numPoints = 10000
x2, y2 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [-4, 5]
cov = [[2, 10], [1, 8]]
numPoints = 10000
x3, y3 = numpy.random.multivariate_normal(mean, cov, numPoints).T

plt.plot(x1, y1, 'or')
plt.show
plt.plot(x2, y2, 'og')
plt.show
plt.plot(x3, y3, 'og')
plt.title('Non-Linear Separability')
plt.show()

#Highly Correlated
mean = [0, -1]
cov = [[5, 4], [1, 1]]
numPoints = 10000
x1, y1 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [10, 10]
cov = [[5, 4], [1, 1]]
numPoints = 10000
x2, y2 = numpy.random.multivariate_normal(mean, cov, numPoints).T

plt.plot(x1, y1, 'or')
plt.show
plt.plot(x2, y2, 'og')
plt.title('Hightly Correlated Features')
plt.show()

#Multi Modal
mean = [8, 8]
cov = [[1, 3], [3, 1]]
numPoints = 10000
x1, y1 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [-8, 8]
cov = [[1, 3], [3, 1]]
numPoints = 10000
x2, y2 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [-8, -8]
cov = [[1, 3], [3, 1]]
numPoints = 10000
x3, y3 = numpy.random.multivariate_normal(mean, cov, numPoints).T

mean = [8, -8]
cov = [[1, 3], [3, 1]]
numPoints = 10000
x4, y4 = numpy.random.multivariate_normal(mean, cov, numPoints).T

plt.plot(x1, y1, 'or')
plt.show
plt.plot(x2, y2, 'og')
plt.show
plt.plot(x3, y3, 'or')
plt.show
plt.plot(x4, y4, 'og')
plt.show
plt.title('Multi Modal Features')
plt.show()