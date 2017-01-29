

class Relu:
    def __init__(self):
        self.mask = None

    # xがバッチデータ(配列)のことも考慮してmaskに保存する
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        # 0以下は0にして返す
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 0以下は0にして返す
        dout[self.mask] = 0
        return dout
