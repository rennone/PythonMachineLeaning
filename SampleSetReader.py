import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


def ReadIris():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    return X, y

def ToStd(X):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def ReadStdIrisTrainTest():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # テストデータとトレージングデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
    sc = StandardScaler()
    # トレーニングデータの平均と標準偏差を計算
    sc.fit((X_train))
    # 標準化
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test
