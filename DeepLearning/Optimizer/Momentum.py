import numpy as np


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr_ = lr
        self.momentum_ = momentum
        self.v = None

    def update(self, params, grad):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum_ * self.v[key] - self.lr_ * grad[key]
            params[key] += self.v[key]
