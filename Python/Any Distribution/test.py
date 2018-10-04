#Nicholas Klein
#Created 9/26/18, Last edit 9/26/18
#A place to test things I don't understand in pyton

import numpy
import matplotlib.pyplot as plt
import random

def test():
    x = []
    y = [2, 2, 4, 1]
    
    for i in range(4):
        x.append(i)
    
    plt.figure()              #Adds a figure
    plt.stem(x, y)            #Makes a stem plot for the figure
    plt.title('Random numbers for any distribution')    #Creates a title for figure
    plt.show() 
    
test()
    