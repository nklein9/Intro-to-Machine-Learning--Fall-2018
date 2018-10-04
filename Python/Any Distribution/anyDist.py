#Nicholas Klein
#Created 9/26/18, Last edit 9/26/18
#Random Numbers for any distribution for Machine Learning

import numpy
import matplotlib.pyplot as plt
import random

def anyDist (num, containers):
#num is number of units generated, containers is the percentage of units that should fall in an area.
#Containers is a list that must be given in individual percentages from lowest range to highest(.25, .15, .6 would be be 0-25,26-40,41-100).
#Percentages must equal 1
    if (sum(containers) == 1.0):
        random.seed()                       #changes seed based on system clock
        n = []                              #initializes n as a list to place random numbers in
        b = []                              #initializes a numbered list to show containers on plot
        c = []                              #initializes a list to count the number of units in each container.
        cont = len(containers)              #Checks how many containers there are
        
        for h in range(num):                #generates a list of num random numbers from 0 to 1 in n
            n.append(random.random())
        
        for i in range(cont):               #Initializes container list and container counter list, and sets all container counter values to zero
            b.append(i)                     #populates b with a range of intigers to show individual containers
            c.append(0)                     #initializes c with zeros
        
        for j in n:                         #Checks each value in n to see if it belongs in container
            for k in range(cont):           #Checks each container to see if j fits there
                if (k == 0):             
                    if (0 <= j <= containers[k]):    #If the lowest range is being checked, the lower bound is zero
                        c[k] += 1           #The counter for the container is incrimented
                    else:
                        pass
                else:
                    if (sum(containers[0:k]) < j <= sum(containers[0:(k+1)])):
                        #If the lowest bound is not being checked, the lower bound is the range below that being checked.
                        #Sums add lower values to check for range requested
                        c[k] += 1            #The counter for the container is incrimented

                    else:
                        pass
                    
        plt.figure()                        #Adds a figure
        plt.stem(b, c)                      #Makes a stem plot for the figure
        plt.title('Random numbers for any distribution')    #Creates a title for figure
        plt.show()                          #displays the figure
        
    else:                                   #If the containers don't equal 1, it throws an error
        print('Error: The distribution does not sum to one.')
        return 'false'
        

a = 10000
b = [.25, .15, .3, .15, .15]

anyDist(a, b)