# Nicholas Klein
# Python 3.6
# Created 11/4/18, Last edit 11/9/18
# Single Perceptron

# Import Statements
import numpy as np
import numpy.random as nran
import matplotlib.pyplot as plt


def main():
    # create each group and bias, set Y label for each group
    bias = np.ones((1, 200))
    group1 = nran.multivariate_normal([10, 10], [[10, 0], [0, 10]], 100).T
    group2 = nran.multivariate_normal([20, 20], [[10, 0], [0, 10]], 100).T
    y = np.concatenate((-1 * np.ones((1, 100)), np.ones((1, 100))), axis=1)

    # concatanate dataset
    x = np.concatenate((group1, group2), axis=1)
    x = np.concatenate((x, bias), axis=0)
    data = np.concatenate((x, y), axis=0)

    # split data into training and testing
    train_data = data[:, :int(data.shape[1] * 0.7)]
    test_data = data[:, int(data.shape[1] * 0.7):]
    train_x = np.matrix(train_data[:3, :])
    train_y = np.matrix(train_data[3, :])
    test_x = np.matrix(test_data[:3, :])
    test_y = np.matrix(test_data[3, :])

    # gradient decent
    i = 0  # Iteration counter
    e = 25000   # epochs
    w = np.ones((x.shape[0], 1))  # weights
    j = np.ones((x.shape[0], 1))  # dJ/dTheta
    a = 0.0001
    while abs(np.max(j)) > 0.01 and i < e:
        predict = np.multiply(-w.transpose() * train_x, train_y)
        isCorrect = list(map(lambda x: False if x < 0 else True, predict.tolist()[0]))
        X_mismatch = [x for x, c in zip(train_x.transpose().tolist(), isCorrect) if not c]
        Y_mismatch = [y for y, c in zip(train_y.transpose().tolist(), isCorrect) if not c]

        try:  # mislabels exist
            j = np.sum(np.multiply(X_mismatch, Y_mismatch).transpose(), axis=1)
            j = j[:, np.newaxis]
            w -= a * j
            i += 1

        except:  # no mislabels
            break

    if i == e:
        print('Did not converge')

    # Calculate accuracy and plot predictions
    prediction = -w.transpose() * test_x
    print('Prediction: ',prediction)
    prediction = list(map(lambda x: 1 if x > 0 else -1, prediction.tolist()[0]))
    correct = sum(map(lambda p, y: p == y, prediction, test_y.tolist()[0])) / len(prediction) * 100
    print('Accuracy: ',correct)

    pred0 = []
    pred1 = []
    for i in range(test_x.shape[1]):
        if prediction[i] == -1:
            pred0.append(tuple(test_x[:2, i].tolist()))
        elif prediction[i] == 1:
            pred1.append(tuple(test_x[:2, i].tolist()))

    pred0x, pred0y = zip(*pred0)
    pred1x, pred1y = zip(*pred1)

    plt.plot(pred0x, pred0y, '.', color='red')
    plt.plot(pred1x, pred1y, '.', color='green')
    plt.show()

    # graph boundary
    points = []
    for i in range(100):
        for j in range(100):
            points.append([i, j, 1])
    labels = -w.transpose() * np.matrix(points).T
    labels = list(map(lambda x: -1 if x < 0 else 1, labels.tolist()[0]))
    paint = list(map(lambda x, y: (x[0], x[1], x[2], y), points, labels))
    paint0x, paint0y, _, _ = list(zip(*filter(lambda x: x[3] == -1, paint)))
    paint1x, paint1y, _, _ = list(zip(*filter(lambda x: x[3] == 1, paint)))

    plt.plot(paint0x, paint0y, '.', color='red')
    plt.plot(paint1x, paint1y, '.', color='green')
    plt.title('Decision Boundary')
    plt.show()

if __name__ == '__main__':
    main()
