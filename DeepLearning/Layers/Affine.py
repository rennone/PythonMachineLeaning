import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
        pass

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)

        self.db = np.sum(dout, axis=0)  # p151参照

        # 入力データの形状を戻す
        dx = dx.reshape(*self.original_x_shape)
        return dx
