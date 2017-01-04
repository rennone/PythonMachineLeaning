# ゼロから作るDeepLearning参照

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # x,yが入力された時の出力
    # バッチデータの場合x,yが配列のこともある
    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    # 出力がdoutだった場合, δf/δx, δf/δyを返す
    def backward(self, dout):
        # x, yはひっくり返す
        # p137参照
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
