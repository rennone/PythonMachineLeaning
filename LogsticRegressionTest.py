import numpy as np
import matplotlib.pyplot as plt
import random
from make_data import make_data_plane
from LogisticRegression import LogisticRegression

X, T = make_data_plane(100)
lr = LogisticRegression(eta=0.1, n_iter=100)
lr.fit(X,T)

# 図を描くための準備
seq = np.arange(-3, 3, 0.01)
xlist, ylist = np.meshgrid(seq, seq)
zlist = [ lr.activation(np.array([x,y]).T) for x, y in zip(xlist, ylist) ]

# 分離平面と散布図を描画
plt.imshow(zlist, extent=[-3,3,-3,3], origin='lower', cmap=plt.cm.PiYG_r)
plt.plot(X[T == 1, 0], X[T == 1, 1], 'o', color='red')
plt.plot(X[T == 0, 0], X[T == 0, 1], 'o', color='blue')

plt.show()
