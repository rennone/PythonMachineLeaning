

# W <- W - η dL / dW
class SGD:
    def __init__(self, lr=0.01):
        self.lr_ = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr_ * grads[key]
