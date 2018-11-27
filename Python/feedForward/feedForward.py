# Nicholas Klein
# Python 3.6
# Created 11/9/18, Last edit 11/12/18
# Feed Forward Neural Network

# Used example code from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#   By Jason Brownlee

if __name__ == '__main__':

    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    import numpy as np
    import pandas
    # fix random seed for reproducibility
    np.random.seed(7)

    # load pima indians dataset
    dataset = pandas.read_csv('pima-indians-diabetes.csv')
    # split into input (X) and output (Y) variables
    X = dataset.iloc[:,0:8]
    Y = dataset.iloc[:,8]

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
