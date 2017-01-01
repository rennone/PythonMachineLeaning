import numpy as np


# シグモイド関数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


# ステップ関数
def step(x):
    y = x > 0
    return y.astype(np.int)


# Relu関数
def relu(x):
    return np.max(0, x)


# 恒等関数
def identity(x):
    return x

# オーバーフロー対策に関しては
# ゼロから作るDeepLearningのp69参照
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

