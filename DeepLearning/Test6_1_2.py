import numpy as np
from DeepLearning.TwoLayerNet import TwoLayerNet

from DataSet.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

import matplotlib.pyplot as plt
def learn(optimizer):
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.gradient(x_batch, t_batch)
        optimizer.update(net.params, grad)

        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.show()


def test(optimizer):
    params = {}
    params['x'] = -7.0
    params['y'] = 2.0

    def f(x, y):
        return x ** 2 / 20.0 + y ** 2

    def gradient(params):
        grad = {}
        grad['x'] = 0.1 * params['x']
        grad['y'] = 2 * params['y']
        return grad

    xhist = []
    yhist = []
    for i in range(30):
        xhist.append(params['x'])
        yhist.append(params['y'])
        grad = gradient(params)
        optimizer.update(params, grad)
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # for simple contour line
    mask = Z > 7
    Z[mask] = 0

    # plot
    plt.plot(xhist, yhist, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

# 6.1.2 SGD -----------------
from DeepLearning.Optimizer.SGD import SGD
# learn(optimizer=SGD())
#test(optimizer=SGD(lr=0.95))

# 6.1.4 Momentum
from DeepLearning.Optimizer.Momentum import Momentum
# earn(optimizer=Momentum())
# test(optimizer=Momentum(lr=0.1))

from DeepLearning.Optimizer.AdaGrad import AdaGrad
test(optimizer=AdaGrad(lr=1.5))