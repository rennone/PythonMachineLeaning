import numpy as np

class AdaGrad:
    def __init__(self, lr):
        self.lr_ = lr
        self.h_ = None

    def update(self, params, grad):
        if self.h_ is None:
            self.h_ = {}
            for key, val in params.items():
                self.h_[key] = np.zeros_like(val)

        for key in params.keys():
            self.h_[key] += grad[key] * grad[key]
            params[key] -= self.lr_ * grad[key] / (np.sqrt(self.h_[key]) + 1e-7)