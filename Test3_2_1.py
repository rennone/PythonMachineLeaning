from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels:", np.unique(y))

# テストデータとトレージングデータに分割
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# トレーニングデータの平均と標準偏差を計算
sc.fit((X_train))

# 標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)

ppn.fit(X_train_std,y_train)

# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)

# 誤サンプルの数を表示
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import  accuracy_score

print('Accurancy: %.2f' % accuracy_score(y_test, y_pred))

from plot_decision_regions import plot_decision_regions

# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))

# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))

import matplotlib.pyplot as plt

plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()
