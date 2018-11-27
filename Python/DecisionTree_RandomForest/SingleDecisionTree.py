# Nicholas Klein
# Python 3.6
# Created 11/20/18, Last edit 11/26/18
# Single Decision Tree

if __name__ == '__main__':
    import pandas
    import numpy as np
    from sklearn import model_selection
    from sklearn.tree import DecisionTreeClassifier

    # Seed
    seed = np.random.seed(7)

    # load pima indians dataset
    DIR = r"C:/Users/nklei/Documents/Homework/ML/Python/DecisionTree_RandomForest/"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(DIR + 'pima-indians-diabetes.csv', names=names)

    # Build model
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    max_features = 3
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = DecisionTreeClassifier()

    # Test model
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    accuracy = results.mean() * 100
    print("Accuracy: ", accuracy)
