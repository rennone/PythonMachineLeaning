import numpy as np
from Functions.functions import softmax, cross_entropy_error, gradient_descent, gradient


class simpleNet:
    def __init__(self):
        # ガウス分布で初期化
        # self.W = np.random.randn(2,3)
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094],
                           [0.85557411, 0.03563661, 0.69422093]])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

np.random.seed(0)
net = simpleNet()

print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0,0,1])
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)

dW = gradient(f, net.W)
print(dW)
