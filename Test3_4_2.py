from sklearn.svm import SVC

from DataSet.iris import learn_iris

# ---------------------
# 線形SVMのインスタンスを生成
svm = SVC(kernel='linear', C=1.0, random_state=0)
# irisデータに対して学習
learn_iris(svm, title='SVM')


# ---------------------
from sklearn.linear_model import SGDClassifier


# 確率的勾配降下法バージョンのパーセプトロン
ppn = SGDClassifier(loss='perceptron')
learn_iris(ppn, title='Perceptron')

# 確率的勾配降下法バージョンのロジスティック回帰
lr = SGDClassifier(loss='log')
learn_iris(lr, title='LogisticRegression')

# 確率的勾配降下法バージョンのSVN(損失関数=ヒンジ関数)
svm = SGDClassifier(loss='hinge')
learn_iris(svm, title='SVM2')


