import pandas as pd

import numpy as np

def pandasminus():
    print('fefe')


def onehot2cat(labels, mat):
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
    print(catslist)

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

        # print(currline)
        # finaloutput.append(np.ndarray(currline))
        finaloutput.loc[finaloutput.shape[0]] = currline

        # cnt = cnt + 1
        # if cnt > 20 :
        #     break
    finaloutput =finaloutput.drop_duplicates(keep=False)
    finaloutput.to_csv('data/adult_generated.data',index_label=False,index=False)



def exe_once_for_writing():
    headers = ['age', 'hworkclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'class']

    # Read in the CSV file and convert "?" to NaN

    df = pd.read_csv('data/adult.data',header=None, names=headers, na_values="?")
    df = df.select_dtypes(include=['object'])
    df.to_csv('data/adult_processed_data',index_label=False,index=False)



def main():
    print('here')
    # original = pd.read_csv('data/test2.csv', float_precision='%.3f')

    headers = ['age', 'hworkclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain',
             'capital-loss', 'hours-per-week', 'native-country', 'class']

    # Read in the CSV file and convert "?" to NaN

    df = pd.read_csv('data/adult.data',header=None, names=headers, na_values="?")
    df = df.select_dtypes(include=['object'])
    print(df.head())
    # df = df.apply(lambda x: x.astype('category'))

    df = df.loc[df['class'] == ' <=50K'].iloc[:, :-1]
    df_onehot = pd.get_dummies(df)
    print(df_onehot.head())
    print(df_onehot.dtypes)


    # dfpost = pd.read_csv('data/test4.csv',header=None, na_values="?")

    dfpost = pd.read_csv('data/adult_mid_exclude_class_ +50K.csv', header=None, na_values="?")
    onehot2cat(list(df_onehot.columns.values), dfpost.values)
    # #
    # #
    # res = gumbel_softmax(df_onehot.values, 0.2, hard=False)
    # sess = tf.Session()
    # res = sess.run(res)
    # print(res)
    # res = pd.DataFrame(res)
    # gan.gan_org(res, splitpoint=0.9)


    # for i in range(10):
    #     print(sum(res[i]))



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
    main()