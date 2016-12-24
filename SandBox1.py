# -*- coding: utf-8 -*-


class Hoge:
    def __init__(self, eta):
        self.eta = eta


class Foo(Hoge):
    def __init__(self, eta):
        Hoge.__init__(self, eta)

    def P(self):
        print(self.eta)

a = Foo(10)

a.P()
