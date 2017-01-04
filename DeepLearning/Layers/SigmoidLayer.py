import numpy as np

class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out