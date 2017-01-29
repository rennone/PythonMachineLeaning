import numpy as np
from Functions.functions import softmax, cross_entropy_error

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    # softmaxは最後のレイヤなのでdout = 1となっている
    # batch_sizeで割るのが謎, p156ではデータ1つ当たりの誤差を伝搬させるためとあるが
    # yの各要素の値は 0 ~ 1の範囲(0 ~ batch_sizeではない)なので, batch_sizeで割ると大きさがずれるのでは?
    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # 教師データがone-hot-vectorの場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        # インデックス指定の場合
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
