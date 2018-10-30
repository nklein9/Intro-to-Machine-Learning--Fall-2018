#Nicholas Klein
#Created 10/8/18, Last edit 10/29/18
#Bayes Theorem

import numpy as np
import matplotlib.pyplot as plt
import random as ran
import math

def main():
    #Declare lists
    group1 = []
    group2 = []
    lik1 = []
    lik2 = []
    evidence = []
    
    # Generate Random Variables
    ran.seed()                       #changes seed based on system clock
    
    a = ran.random()
    prior1 = a
    prior2 = 1 - prior1
    group1 = genRandom(round(prior1*100), 10, 0)
    group2 = genRandom(round(prior2*100), 10, 5)
    
    #calculate likelyhood and evidence
    lik1 = calcLikelyhood(group1)
    lik2 = calcLikelyhood(group2)
    evidence = calcEvidence(group1, group2)

    #calculate Joint probability

    #plot everything
    plt.figure()                                                                #Adds a figure
    plt.scatter(group1, lik1, c = 'b')                                          #Scatter of group 1 likelihood
    plt.scatter(group2, lik2, c = 'r')                                          #Scatter of group 2 likelihood
    plt.legend()
    plt.title("Likelyhood")                                                     #Creates a title for figure
    plt.show()                                                                  #displays the figure

def calcProb(lik, prior, evidence):
    P = np.zeros(16)
    for i in lik:
        P[i] = (lik[i] * prior) / evidence[i]
    return P

def calcEvidence(groupa, groupb):
    bigGroup = []
    evidence = np.zeros(16)
    for i in groupa:
        bigGroup.append(i)
    for i in groupb:
        bigGroup.append(i)
    evidence = calcLikelyhood(bigGroup)
    
    return evidence
        
def genRandom(size, multiplier, shift):
    ra = []
    for i in range(size):
        h = ran.random()
        ra.append((multiplier * h) + shift)
        
    return ra

def calcLikelyhood(group):
    count = np.zeros(16)                                        #counter to track objects in each container
    lik = []                                                    #likelyhood an object in group is in each container
    for i in group:
        if (0 <= i <= 1):
            count[0] += 1
        if (1 < i <= 2):
            count[1] += 1
        if (2 < i <= 3):
            count[2] += 1
        if (3 < i <= 4):
            count[3] += 1
        if (4 < i <= 5):
            count[4] += 1
        if (5 < i <= 6):
            count[5] += 1
        if (6 < i <= 7):
            count[6] += 1
        if (7 < i <= 8):
            count[7] += 1
        if (8 < i <= 9):
            count[8] += 1
        if (9 < i <= 10):
            count[9] += 1
        if (10 < i <= 11):
            count[10] += 1
        if (11 < i <= 12):
            count[11] += 1
        if (12 < i <= 13):
            count[12] += 1
        if (13 < i <= 14):
            count[13] += 1
        if (14 < i <= 15):
            count[14] += 1
        if (15 < i <= 16):
            count[15] += 1
        else:
            break
    for i in count:
        lik.append(i/len(group))
    return lik

main()