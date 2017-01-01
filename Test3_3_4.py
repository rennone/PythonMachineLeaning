import matplotlib.pyplot as plt

from DataSet.iris import learn_iris
from LogisticRegression import LogisticRegression

# 独自ロジスティック回帰のテスト
lr = LogisticRegression(n_iter=15,eta=0.01,random_state=1)

# 学習させる
learn_iris(classifier=lr, title='LogisticRegression')

# コスト量の遷移グラフ
plt.plot(range(1, len(lr.cost_)+1), lr.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Average Cost")
plt.show()