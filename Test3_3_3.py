from sklearn.linear_model import LogisticRegression
from SampleSetReader import ReadStdIrisTrainTest
from plot_decision_regions import plot_decision_region_iris
import numpy as np


X_train_std, X_test_std, y_train, y_test = ReadStdIrisTrainTest()
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))

# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

plot_decision_region_iris(X_combined_std, y_combined, classifier=lr, test_idx = range(105,150))

print(lr.predict_proba(X_test_std[0,:]))

