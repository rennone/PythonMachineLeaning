from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
        self.eta = eta
        self.n_iter = n_iter

        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            # シャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weight(xi, target))
            # 平均コストの計算
            avg_cost = sum(cost)/len(y)
            # 平均コスト
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self,X,y):
        # 初期化チェック
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        # 目的変数yの要素数が2以上の場合は各サンプルの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1 :
            for xi, target in zip(X,y):
                self._update_weight(xi, target)
        # 目的変数yの要素数が1の場合はサンプル全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weight(X, y)
        return self

    def _update_weight(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        # 重み(w1~)の更新
        self.w_[1:] += self.eta * xi.dot(error)
        # 重みw0の更新
        self.w_[0] += self.eta * error
        cost = 0.5*error**2
        return cost

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return  X[r], y[r]

    # 重みの初期化
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) > 0.0, 1, -1)



