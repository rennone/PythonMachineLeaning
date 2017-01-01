import numpy as np
import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions


def read_xor(data_num=200):
    np.random.seed(0)

    # 標準正規分布に従う乱数で200行2列の行列を生成
    x_xor = np.random.randn(data_num, 2)
    # 2つの引数に対してxorを実行
    y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)

    # true = 1, false = 0とする
    y_xor = np.where(y_xor, 1, -1)
    return x_xor, y_xor


def plot_xor(data_num=200):
    x_xor, y_xor = read_xor(data_num)

    # 1=blue x
    plt.scatter(x_xor[y_xor == 1, 0], x_xor[y_xor == 1, 1], c='b', marker='x', label='1')

    # -1=red rectangle
    plt.scatter(x_xor[y_xor == -1, 0], x_xor[y_xor == -1, 1], c='r', marker='s', label='-1')

    # 軸の範囲を設定
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])

    # 判例をつける
    plt.legend(loc='best')
    plt.show()


def learn_xor(classifier, data_num=200, title='Result'):
    x_xor, y_xor = read_xor(data_num)
    classifier.fit(x_xor, y_xor)

    plot_decision_regions(x_xor, y_xor, classifier=classifier)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()
