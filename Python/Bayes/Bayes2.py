#Nicholas Klein
#Created 10/29/18, Last edit 10/29/18
#Bayes Theorem

#Import Statements
import numpy as np
import matplotlib.pyplot as plt
import random as ran

#define functions
def separateData(dataIN, trainPer):
    #Takes in a data set and separates into training and testing data
    training = []
    testing = []
    dataOUT = []
    for i in range(len(dataIN)):
        if (i < 0.01*trainPer*len(dataIN)):
            training.append(dataIN[i])
        else:
            testing.append(dataIN[i])
    dataOUT = (training, testing)
    return dataOUT


def calcEvidence(groupa, groupb):
    bigGroup = []
    evidence = np.zeros(16)
    for i in groupa:
        bigGroup.append(i)
    for i in groupb:
        bigGroup.append(i)
    evidence = calcLikelyhood(bigGroup)

    return evidence

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

def bayes(lik, prior, ev):
    b = []
    for i in range(len(lik)):
        b.append(lik[i]*(prior/ev[i]))
    return b

def main():
    #declare lists
    group1 = []
    group2 = []
    data1 = []
    data2 = []
    train1 = []
    test1 = []
    train2 = []
    test2 = []

    #generate groups
    group1 = np.random.normal(5, 5, 1000)
    group2 = np.random.normal(8, 5, 1000)

    #create data
    data1 = separateData(group1, 70)
    data2 = separateData(group2, 70)
    train1 = data1[0]
    test1 = data1[1]
    train2 = data2[0]
    test2 = data2[1]

    #gather Bayes information
    prior1 = ran.random()
    prior2 = 1 - prior1
    likTr1 = calcLikelyhood(train1)
    likTr2 = calcLikelyhood(train2)
    ev = calcEvidence(train1, train2)
    likTe1 = calcLikelyhood(test1)
    likTe2 = calcLikelyhood(test2)
    ev2 = calcEvidence(test1, test2)

    bayesTr1 = bayes(likTr1, prior1, ev)
    bayesTr2 = bayes(likTr2, prior2, ev)

    bayesTe1 = bayes(likTe1, bayesTr1, ev2)
    bayesTe2 = bayes(likTe2, bayesTr2, ev2)

    #plots
    plt.figure()  # Adds a figure
    plt.subplot(2,1,1)
    plt.bar(range(16), bayesTr1, color = 'b')
    plt.bar(range(16), bayesTr2, color = 'r')
    plt.ylabel("Training")  # Creates a title for figure

    plt.subplot(2, 1, 2)
    plt.bar(range(16), bayesTe1, color = 'b')
    plt.bar(range(16), bayesTe2, color = 'r')
    plt.ylabel("Testing")  # Creates a title for figure
    plt.show()  # displays the figure


#Run the program
if __name__ == '__main__':
    main()