import numpy as np


# シグモイド関数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# シグモイド関数の微分形
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

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
# p118で使っているsoftmaxは以下のように変更されている
def softmax(x):
    # バッチ処理用
    if x.ndim == 2:
        # 転置させないと, np.maxの次元数が想定通りにならない
        x = x.T
        x = x - np.max(x, axis=0)  # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


# 平均二乗誤差
def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# バッチデータ対応版
# 交差エントロピー誤差
# y : 1次元 or 2次元配列(バッチ処理の時)の予測データ
# t : 教師データ
def cross_entropy_error(y, t):
    # 1次元データだった場合バッチ数1の2次元データ(バッチデータ)に変更
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合,正解ラベルのインデックスに変換
    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # ラベル表現(2など,正解のインデックスが直接指定されている)の場合
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size;


# 数値微分(中心差分を用いた微分)
def diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 偏微分
def partial_diff(f, x, idx):
    h = 1e-4
    val = x[idx]

    # f(x+h)の計算
    x[idx] = val + h
    fxh1 = f(x)

    # f(x-h)の計算
    x[idx] = val - h
    fxh2 = f(x)

    # 計算
    ret = (fxh1 - fxh2) / (2*h)

    # 元に戻しておく
    x[idx] = val

    return ret


# 勾配計算
# ゼロから作るDeepLearning p112のコラム参照
def gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    # xが多次元配列でも全要素にアクセスかつ書き換え可能なようにイテレータを使う
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = gradient(f, x)

        x -= lr*grad

    return x