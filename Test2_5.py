from AdalineGD import AdalineGD
import matplotlib.pyplot as plt
import numpy as np
from SampleSetReader import ReadIris


def calc2_5(X, y, ax, eta):
    # 勾配降下法によるAdalineの学習
    adal = AdalineGD(n_iter=10, eta=eta).fit(X, y)
    # エポック数とコストの関係を表す折れ線グラフのプロット(縦軸のコストは常用対数)
    ax.plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
    # ラベル設定
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(sum-squared-error')
    # タイトル
    ax.set_title('Adaline-Learning rate' + str(eta))


def Test2_5(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(8,4))

    calc2_5(X, y, ax[0], 0.01)
    calc2_5(X, y, ax[1], 0.0001)
    plt.show()

X, y = ReadIris()

Test2_5(X,y)

