#Nicholas Klein
#Created 10/4/18, Last edit 10/9/18
#Linear Regression Algorithm using firefire data

#Parts of main adapted from: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    # Code source: Jaques Grobler
    # License: BSD 3 clause
    
#For expected error failure: https://github.com/scikit-learn/scikit-learn/issues/12226

import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def main():
    #Pull Temperature and Area burned data
    data = withdrawCSVData('forestfires.csv', 8, 12)
    tempTraining = data[0]
    areaTraining = data[1]
    tempTesting = data[2]
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
    plt.scatter(tempTesting, areaTesting,  color='black')
    plt.plot(tempTesting, areaPredict, color='blue', linewidth=3)
    
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
    
def withdrawCSVData(csv, r1, r2):                       #pulls data from csv's row r1 and row r2 and returns a matrix
    with open('{}'.format(csv)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        a = []
        b = []
        ar = []
        br = []
        ae = []
        be = []
        for row in csv_reader:
            a = row [r1]
            b = row [r2]
        
        lar = 0.7*len(a)
        lbr = 0.7*len(b)
        lae = 0.3*len(a)
        lbe = 0.3*len(b)
        
        ar = a[0:lar]                                   #Builds Training Data
        br = b[0:lbr]
        ae = a[0:lae]                                   #Builds Testing Data
        be = b[0:lbe]
    data = (ar, br, ae, be)
    return data
 