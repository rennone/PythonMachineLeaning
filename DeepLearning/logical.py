import numpy as np

# 論理回路
# x1*w1 + x1*w2 + b <= 0  → 0
# x1*w1 + x1*w2 + b  > 0  → 1
def logical(x, w, b):
    return 0 if np.sum(x*w)+b <= 0 else 1


def AND(x1, x2):
    return logical(x=[x1,x2], w=[0.5,0.5], b=-0.7)


def NAND(x1, x2):
    return logical(x=[x1, x2], w=[-0.5, -0.5], b=0.7)


def OR(x1, x2):
    return logical(x=[x1, x2], w=[0.5, 0.5], b=-0.2)


def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

