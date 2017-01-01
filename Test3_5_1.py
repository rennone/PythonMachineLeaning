from sklearn.svm import SVC

from DataSet.iris import learn_iris
from DataSet.xor import learn_xor

# SVMに関するテスト


# カーネルトリック

# カーネル = 簡単に言うと, 2つのサンプル(x_i, x_j)間の類似度を表す関数( 0 ~ 1 )
# ガウスカーネル
# k(x_i, x_j) = exp(- |x_i - x_j|^2 / 2σ^2)
#             = exp(-γ|x_i - x_j|^2)  (γ = 1/2σ^2)


svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
learn_xor(classifier=svm)
learn_iris(classifier=svm, title='SVM')

# --------------------
# Γを大きくした場合 → 決定境界が複雑になりか学習が発生する
svm = SVC(kernel='rbf', random_state=0, gamma=100, C=10.0)
learn_iris(classifier=svm, title='large Gamma')


