
import numpy as np
import matplotlib.pyplot as plt
import random

# データ点の個数
from xlwings import xrange

N = 100

# データ点のために乱数列を固定
np.random.seed(0)

# ランダムな N×2 行列を生成 = 2次元空間上のランダムな点 N 個
X = np.random.randn(N, 2)

def h(x, y):
    return 5 * x + 3 * y - 1  #  真の分離平面 5x + 3y = 1

T = np.array([1 if h(x, y) > 0 else 0 for x, y in X])

# シグモイド関数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 特徴関数
def phi(x, y):
    return np.array([x, y, 1])


np.random.seed() # 乱数を初期化
w = np.random.randn(3)
eta = 0.1

for i in xrange(50):
    list_ = list(range(N))
    random.shuffle(list_)

    misses = 0 # 予測を外した回数
    for n in list_:
        x_n, y_n = X[n, :]
        t_n = T[n]

        # 予測
        feature = phi(x_n, y_n)
        predict = sigmoid(np.inner(w, feature))

        w -= eta * (predict - t_n) * feature

    eta *= 0.9



# 図を描くための準備
seq = np.arange(-3, 3, 0.01)
xlist, ylist = np.meshgrid(seq, seq)
zlist = [sigmoid(np.inner(w, phi(x, y))) for x, y in zip(xlist, ylist)]

# 分離平面と散布図を描画
plt.imshow(zlist, extent=[-3,3,-3,3], origin='lower', cmap=plt.cm.PiYG_r)
plt.plot(X[T== 1,0], X[T== 1,1], 'o', color='red')
plt.plot(X[T== 0,0], X[T== 0,1], 'o', color='blue')

plt.show()