#Nicholas Klein
#Created 9/27/18, Last edit 10/4/18
#Linear Regression Algorithm for Machine Learning

import numpy as np
import matplotlib.pyplot as plt
import random as ran
import decimal
import math

def linReg(num, m, b):
    ran.seed()                       #changes seed based on system clock
    
    x = []
    y = []
    y1 = []
    h = []
    j = []
    
    theta0 = 0                              #initial Y intercept guess of 0
    theta1 = 0                              #initial Slope guess of 0
    alpha = 0.0001                            #Learning incriment
    jrun = 10                               #how many times cost function runs
    
    x = ran.sample(range(100), num)

    for i in x:
        y.append(m*i + b + ran.randrange(-10,10,1))
        y1.append(m*i + b)
   
    h = hypothosis(theta0, theta1, x)
    
    
    
    for  i in range(jrun):
        j = costFunc(num, x, h, y, alpha, theta0, theta1)
        
        theta0 = j[0]
        theta1 = j[1]
        
        h = hypothosis(theta0, theta1, x)
        
    pe = percentError(h, y)
    
    tp1 = format(theta1, '.4g')
    tp0 = format(theta0, '.4g')
    plt.figure()                                                        #Adds a figure
    plt.plot(x, y1, 'r', label= 'y={}x+{}'.format(m, b))                #Plots true line
    plt.plot(x, h, 'g--', label='h={}x+{}'.format(tp1, tp0))            #plots hypothesis
    plt.scatter (x, y, c = 'b')                                         #Scatter of noise values
    plt.legend()
    plt.text(60, .025, r'%Error: {}'.format(pe))
    plt.title('Linear Regression')                       #Creates a title for figure
    plt.show()                          #displays the figure
    
def hypothosis(theta0, theta1, x):
    h = []
    for i in x:
        h.append(theta0 + (theta1 * i))
    
    return h
    
def costFunc(n, x, h, y, a, t0, t1):
    co = []
    f = []
    
    for i in range(n):
        co.append(decimal.Decimal((h[i]-y[i])))
    for i in range (n):
        f.append(x[i] * co[i])
    tn0 = t0 - decimal.Decimal(a) * sum(co)/n
    tn1 = t1 - decimal.Decimal(a) * sum(f)/n
    
    print ("Sum co: ", sum(co))
    print ("sum f: ", sum(f))
    print ("t0: ", t0)
    print ("t1: ", t1)
    print ("tn0: ", tn0)
    print ("tn1: ", tn1)
    
    j = (tn0, tn1)
    
    return j

def percentError(ex, theo):
    h=[]
    for i in range(len(ex)):
        h.append(np.absolute((ex[i] - theo[i])/theo[i]) * 100)
    
    a=sum(h)/len(h)
    pe= format(a, '.4g')
    
    return pe


linReg(100, 2, 1)