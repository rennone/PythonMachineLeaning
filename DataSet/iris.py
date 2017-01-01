import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt


def read_iris():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    x = df.iloc[0:100, [0, 2]].values
    return x, y


def plot_decision_regions_iris(X, y, classifier, test_idx=None,resolution=0.02):
    plot_decision_regions(X, y, classifier=classifier, test_idx=test_idx, resolution=resolution)
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.show()


def to_std(X):
    x_std = np.copy(X)
    x_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    x_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return x_std


def read_std_train_test_data_iris():
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
    return X_train_std, y_train, X_test_std, y_test


# irisのデータに対して学習を行う
# classifier : svm, logistic回帰などの実装した学習クラス
def learn_iris(classifier, title='Result'):
    # データの読み込み
    x_train, y_train, x_test, y_test = read_std_train_test_data_iris()

    # 学習
    classifier.fit(x_train, y_train)

    # トレーニングデータとテストデータの特徴量を行方向に結合
    x_combined = np.vstack((x_train, x_test))

    # トレーニングデータとテストデータのクラスラベルを結合
    y_combined = np.hstack((y_train, y_test))

    plt.title(title)
    plot_decision_regions_iris(x_combined, y_combined, classifier=classifier, test_idx=range(105, 150))

