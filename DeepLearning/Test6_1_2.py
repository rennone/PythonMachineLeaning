import numpy as np
from DeepLearning.TwoLayerNet import TwoLayerNet
from DeepLearning.Optimizer.SGD import SGD

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

# 6.1.2 SGD -----------------
learn(optimizer=SGD())

# 6.1.4 Momentum
from DeepLearning.Optimizer.Momentum import Momentum
learn(optimizer=Momentum())