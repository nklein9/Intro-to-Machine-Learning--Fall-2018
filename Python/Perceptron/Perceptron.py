# Nicholas Klein
# Python 3.6
# Created 11/4/18, Last edit 11/4/18
# Single Perceptron

# Import Statements
import numpy as np
import numpy.random as nran
import matplotlib.pyplot as plt


def predict(inputs, weights):   # Predicts the state of the data based on inputs and weight
    activation=0.0
    for i, w in zip(inputs, weights):
        activation += i*w
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0


def split_data(data):
    train_data = data[:, :int(data.shape[1] * 0.7)]
    test_data = data[:, int(data.shape[1] * 0.7):]
    train_x = np.matrix(train_data[:3, :])
    train_y = np.matrix(train_data[3, :])
    test_x = np.matrix(test_data[:3, :])
    test_y = np.matrix(test_data[3, :])

    new_data = [train_x, train_y, test_x, test_y]
    return new_data


def main():
    # create each group and bias, set Y label for each group
    bias = np.ones((1, 200))
    group1 = nran.multivariate_normal([5, 5], [[5, 0], [0, 5]], 100).T
    group2 = nran.multivariate_normal([10, 10], [[5, 0], [0, 5]], 100).T
    y = np.concatenate((-1 * np.ones((1, 100)), np.ones((1, 100))), axis=1)

    # concatanate dataset
    x = np.concatenate((group1, group2), axis=1)
    x = np.concatenate((bias, x), axis=0)
    data = np.concatenate((x, y), axis=0)

    # split data into training and testing
    split_data(data)

    plt.plot(group1[0], group1[1], '.', color='green')
    plt.plot(group2[0], group2[1], '.', color='red')
    plt.show()


if __name__ == '__main__':
    main()
