# ---http://pbpython.com/categorical-encoding.html


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import numpy as np


def test_the_models(df):

    # print(df.head())

    # print(df.dtypes)
    # df = df.apply(lambda x: x.astype('category'))
    # print(df.dtypes)

    # return
    #
    #
    # print(df.iloc[:,-1])
    #
    # labels = df.iloc[:,-1]
    #
    # X, y = df.iloc[:,0:-1].values, df.iloc[:,-1].values
    #
    # print('X is ')
    # print(X)
    #
    # print('y is ')
    # print(y)

    le = preprocessing.LabelEncoder()
    dfdf = df.apply(le.fit_transform)
    #
    # # le.fit(df)
    # print(dfdf.head())
    # print(dfdf.describe())
    # print('after encoded')
    # print(dfdf.iloc[:,-1].values)


    # dfdfdecoded = le.inverse_transform(dfdf.iloc[:,-1].values)
    # print(dfdfdecoded)


    X, y = dfdf.iloc[:,0:-1].values, dfdf.iloc[:,-1].values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)

    print('y_test')
    print(y_test)

    # classifier = tree.DecisionTreeClassifier()
    # classifier = RandomForestClassifier()
    # classifier = SVC(kernel="linear", C=0.025)
    # classifier = SVC(gamma=2, C=1)
    # classifier = SVC()
    # classifier = MLPClassifier(alpha=1, hidden_layer_sizes = 1000)
    # classifier = MLPClassifier(alpha=1)
    # classifier = MLPClassifier()
    # classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    # classifier = AdaBoostClassifier()
    # classifier = QuadraticDiscriminantAnalysis()
    # classifier = KNeighborsClassifier(6)
    # classifier = classifier.fit(X_train, y_train)


    # classifiers = [
    #     KNeighborsClassifier(3),
    #     # SVC(kernel="linear", C=0.025),
    #     # SVC(gamma=2, C=1),
    #     # GaussianProcessClassifier(1.0 * RBF(1.0)),
    #     DecisionTreeClassifier(max_depth=5),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     # MLPClassifier(alpha=1),
    #     AdaBoostClassifier(),
    #     # GaussianNB(),
    #     QuadraticDiscriminantAnalysis()]

    classifiers = {'KNN3':KNeighborsClassifier(3),
                   # 'Guassian Process 1.0 * RBF(1.0)': GaussianProcessClassifier(1.0 * RBF(1.0)),
                   'DecisionTree':DecisionTreeClassifier(max_depth=5),
                   'RandomForest':RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                   'AdaBoost':AdaBoostClassifier(),
                   'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
                   'MLPClassification, alpha = 1': MLPClassifier(alpha=1),
                   # 'SCV linear':SVC(kernel="linear", C=0.025)
                   }


    res = ''


    for entry in classifiers:
        print(entry)
        classifier = classifiers[entry]


        kfold = StratifiedKFold(n_splits=10,
                                random_state=1).split(X_train, y_train)
        scores = []
        for k, (train, test) in enumerate(kfold):
            classifier.fit(X_train[train], y_train[train])
            score = classifier.score(X_train[test], y_train[test])
            scores.append(score)
            print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k + 1,
                                                            np.bincount(y_train[train]), score))

        score = classifier.score(X_test, y_test)
        print(score)
        # print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

        res += entry + ':' + 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)) + '\n'

    print('\n\n\n-----------------------\n\n\n')
    print(res)



def main():
    # Define the headers since the data does not have any
    headers = ['age','hworkclass','fnlwgt','education',
               'education-num','marital-status','occupation',
               'relationship','race','sex','capital-gain',
               'capital-loss','hours-per-week','native-country','class']

    # Read in the CSV file and convert "?" to NaN
    df = pd.read_csv('data/adult.data',
                      header=None, names=headers, na_values="?" )
    test_the_models(df)

if __name__ == '__main__':
    main()