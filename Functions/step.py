import numpy as np


# ステップ関数
def step(x):
    y = x > 0
    return y.astype(np.int)