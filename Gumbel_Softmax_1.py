
import numpy as np

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import gandiscrete_3_clmethods1 as gan


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




def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y




def simplyplot(filename):
    plt.style.use('ggplot')
    original = pd.read_csv(filename)
    # original = gd.genran2()




    #----------------------------------------------
    scale = 2.5
    fig, axs = plt.subplots(2, 3, figsize=(9*scale, 3*scale), sharey=True)
    # original.plot.bar(stacked=True)
    # original.plot.hist(stacked=True)
    # ser = pd.Series(original['A'])
    # original['A'].value_counts(sort=False).plot(ax = axs[0,0],kind ='bar')
    # original['B'].value_counts(sort=False).plot(ax = axs[0,1],kind ='bar')
    # original['C'].value_counts(sort=False).plot(ax = axs[0,2],kind ='bar')
    # original['D'].value_counts(sort=False).plot(ax = axs[1,0],kind ='bar')
    # original['E'].value_counts(sort=False).plot(ax = axs[1,1],kind ='bar')
    # original['F'].value_counts(sort=False).plot(ax = axs[1,2],kind ='bar')


    #----------------------------------------------

    original['A'].plot(ax = axs[0,0],kind ='hist')
    original['B'].plot(ax = axs[0,1],kind ='hist')
    original['C'].plot(ax = axs[0,2],kind ='hist')
    original['D'].plot(ax = axs[1,0],kind ='hist')
    original['E'].plot(ax = axs[1,1],kind ='hist')
    original['F'].plot(ax = axs[1,2],kind ='hist')

    fig.suptitle(filename)

    plt.show()
    #----------------------------------------------


    # original.plot(kind = 'density')
    #
    #
    # plt.show()


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
                   # 'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
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




def df1_minus_df2(df1 = 'data/adult_mid_exclude_class_ -=50K.csvfin.csv', df2= 'data/adult_processed_data', finalpath = 'data/adult_mid_exclude_class_ -=50K.csvfin.csvfinal.csv'):
    df1 = pd.read_csv(df1)
    # print(df1.head())
    df2 = pd.read_csv(df2)
    # print(df2.head())

    df2dict = set()
    for index, row in df2.iterrows():
        rowstr = row.to_string(header=False, index=False).split('\n')
        rowstr = [str.strip(x) for x in rowstr]
        rowstr = ','.join(rowstr)
        # rowstr = str[row]

        # print(str(rowstr))
        if rowstr not in df2dict:
            df2dict.add(rowstr)

    finalret = pd.DataFrame(columns=df1.columns.values)
    for index, row in df1.iterrows():
        rowstr = row.to_string(header=False, index=False).split('\n')
        rowstr = [str.strip(x) for x in rowstr]
        rowkey = ','.join(rowstr)

        if rowkey not in df2dict:
            finalret.loc[finalret.shape[0]] = rowstr
        else:
            print('found found found!!!' + rowkey)

    finalret.to_csv(finalpath,index_label=False,index=False)



def onehot2cat(labels, mat, outputfilename, labelstr, labelval):
    cats = {}
    catslist = []
    cutpoints = []

    for i in range(len(labels)):
        label = labels[i]
        cat = label.split('_')[0]
        val = label.split('_')[1]
        if cat not in cats:
            cats[cat] = 1
            catslist.append(cat)
            cutpoints.append(i)
    cutpoints.append(len(labels))
    # print(catslist)

    catslist.append(labelstr)
    finaloutput = pd.DataFrame(columns = catslist)

    cnt = 0
    for line in mat:
        currline = []
        for i in range(len(cutpoints) - 1):
            j = i + 1
            start = cutpoints[i]
            end = cutpoints[j]
            # current_label = labels[i].split('_')[0]
            max = 0
            maxval = 0
            for k in range(start, end):
                if max < line[k]:
                    max = line[k]
                    maxval = labels[k].split('_')[1]
            currline.append(maxval)
        currline.append(labelval)

        # print(currline)
        # finaloutput.append(np.ndarray(currline))
        finaloutput.loc[finaloutput.shape[0]] = currline

        # cnt = cnt + 1
        # if cnt > 20 :
        #     break
    # print(finaloutput.head())
    finaloutput =finaloutput.drop_duplicates(keep='first')
    finaloutput.to_csv(outputfilename,index_label=False,index=False)
    return finaloutput.shape


def main():
    print('game start')
    # original = pd.read_csv('data/test2.csv', float_precision='%.3f')

    headers = ['age', 'hworkclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'class']

    # Read in the CSV file and convert "?" to NaN

    df = pd.read_csv('data/adult.data',header=None, names=headers, na_values="?")
    df = df.select_dtypes(include=['object'])


    dfdict = {}
    classes = df['class'].unique()
    for cls in classes:
        dfdict[cls] = df.loc[df['class'] == cls].iloc[:,:-1]
        print(cls)
        # print(dfdict[cls].head())


        df_onehot = pd.get_dummies(dfdict[cls])
        # print(df_onehot)
        #
        #
        res = gumbel_softmax(df_onehot.values, 0.2, hard=False)
        print('gumbel_softmax completed!')
        sess = tf.Session()
        res = sess.run(res)
        # print(res)
        # for i in range(10):
        #     print(sum(res[i]))
        res = pd.DataFrame(res)

        fnstr = cls
        fnstr = fnstr.replace('>','+')
        fnstr = fnstr.replace('<', '-')

        outputfilename = 'data/adult_mid_exclude_class_' + fnstr + '.csv'
        ganshape = gan.gan_org(res, splitpoint=0.9,outputfilename=outputfilename)
        print('gan completed! ' + '#' + str(ganshape) + '\t' + outputfilename)

        dfpost = pd.read_csv(outputfilename, header=None, na_values="?")
        decodeshape = onehot2cat(list(df_onehot.columns.values), dfpost.values, outputfilename + 'fin.csv', labelstr='class', labelval=cls)
        print('onehot2cat completed! ' + '#' + str(decodeshape) + '\t' + outputfilename)

    print(df.head())
    



    #-----------------------------------------------------------------
    # df_onehot = pd.get_dummies(df)
    # # print(df_onehot)
    # #
    # #
    # res = gumbel_softmax(df_onehot.values, 0.2, hard=False)
    # sess = tf.Session()
    # res = sess.run(res)
    # print(res)
    # res = pd.DataFrame(res)
    # gan.gan_org(res, splitpoint=0.9)
    # -----------------------------------------------------------------

    ####test the sum of a softmax function result
    # for i in range(10):
    #     print(sum(res[i]))

    ####convert each column to category type
    # df = df.apply(lambda x: x.astype('category'))



    # print('original')
  # print(df.values)
  # print('gumbel')
  #
  # # gumbel_softmax(original, 0.2, hard=True)
  # # res = gumbel_softmax(original.iloc[:,0:1].values, 0.2, hard=False)
  # res = gumbel_softmax(df.values, 0.2, hard=False)
  # sess = tf.Session()
  # res = sess.run(res)
  # print(res)

if  __name__ == '__main__':
    # print('>='.replace('>','+'))
    # main()
    # df1_minus_df2(df1='data/adult_mid_exclude_class_ -=50K.csvfin.csv', df2='data/adult_processed_data',
    #               finalpath='data/adult_mid_exclude_class_ -=50K.csvfin.csvfinal.csv')
    # df1_minus_df2(df1='data/adult_mid_exclude_class_ +50K.csvfin.csv', df2='data/adult_processed_data',
    #               finalpath='data/adult_mid_exclude_class_ +50K.csvfin.csvfinal.csv')

    df1 = pd.read_csv('data/adult_mid_exclude_class_ -=50K.csvfin.csvfinal.csv', header=None)
    df2 = pd.read_csv('data/adult_mid_exclude_class_ +50K.csvfin.csv', header=None)
    frames = [df1,df2]
    frames = pd.concat(frames)
    test_the_models(frames)
    df = pd.read_csv('data/adult_processed_data', header=None, na_values="?")
    test_the_models(df)