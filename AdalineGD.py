import numpy as np


class AdalineGD(object):
    def __init__(self, eta=0.01,n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    # X:配列のような構造. shape = [n_sample, n_features]
    # y:配列のような構造. shape = [n_sample]
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            # φ(w^Tx) = w^Tx
            output = self.net_input(X)
            # y^(i) - φ(z^(i))
            errors = y - output
            # w1...wm の更新, X.Tは転置行列
            # η * Σ_i(y^(i) - φ(z^(i))) * xj^(i)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # コスト関数の計算J(w) = 1/2Σ(y^(i) - φ(z^(i)))^2
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return  self

    # 総入力を計算
    # ret = [n_sample] 全サンプルの結果が配列で返る
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 線形活性化関数の出力を計算
    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)