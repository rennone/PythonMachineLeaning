from DataSet.iris import read_iris, plot_decision_regions_iris
from Perceptron import Perceptron

# Perceptronの定義
ppn = Perceptron(eta=0.01)

# irisデータの読み込み
X, y = read_iris()

# 学習
ppn.fit(X,y)

plot_decision_regions_iris(X, y, classifier=ppn)

