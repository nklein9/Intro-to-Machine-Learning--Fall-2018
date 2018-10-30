#Nicholas Klein
#Created 10/4/18, Last edit 10/16/18
#Linear Regression Algorithm using firefire data

#Parts of main adapted from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    # Code source: Jaques Grobler
    # License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sklearn
#import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#print(os.getcwd())

def main():
    #Pull Temperature and Area burned data
    data = withdrawCSVData(os.path.dirname(os.path.realpath(__file__)) + '\\forestfires.csv', 8, 12)
    ttrMatrix1 = np.ones(len(data[0]))
    ttrMatrix2 = [ttrMatrix1, data[0]]
    tempTraining = np.transpose(ttrMatrix2)
    areaTraining = data[1]
    tteMatrix1 = np.ones(len(data[2]))
    tteMatrix2 = [tteMatrix1, data[2]]
    tempTesting = np.transpose(tteMatrix2)
    areaTesting = data[3]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(tempTraining, areaTraining)
    
    # Make predictions using the testing set
    areaPredict = regr.predict(tempTesting)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(areaTesting, areaPredict))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(areaTesting, areaPredict))
    
    # Plot outputs
    tempTesting = list(map(tuple, tempTesting))
    tempTesting0, tempTesting1 = zip(*tempTesting)
    plt.scatter(tempTesting1, areaTesting,  color='black')
    plt.plot(tempTesting, areaPredict, color='blue', linewidth=3)
    plt.ylim(top=120)
    plt.ylabel("Area Burned")
    plt.xlabel("Temperature Outside")
    
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
    
def withdrawCSVData(csv1, r1: int, r2: int):                       #pulls data from csv's row r1 and row r2 and returns a matrix
    with open(csv1, 'rt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        #line_count = 0
        a = []
        b = []
        ar = []
        br = []
        ae = []
        be = []
        for row in csv_reader:
            a.append(row[r1])
            b.append(row[r2])
        
        lar = int(0.7*(len(a)-1))                  #use 70% to train
        lbr = int(0.7*(len(b)-1))
        lae = int(0.3*(len(a)-1))                 #use 30% to test
        lbe = int(0.3*(len(b)-1))
        
        ar = a[1:lar]                             #Builds Training Data
        ar = list(map(float, ar))
        br = b[1:lbr]
        br = list(map(float, br))
        ae = a[lar+1:lar+lae]  
        ae = list(map(float, ae))                                 #Builds Testing Data
        be = b[lbr+1:lbr+lbe]
        be = list(map(float, be))
    data = (ar, br, ae, be)
    return data

if __name__ == "__main__":
    main()
