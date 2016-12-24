from AdalineGD import AdalineGD
import matplotlib.pyplot as plt
import numpy as np
from SampleSetReader import ReadIris
from plot_decision_regions import plot_decision_regions


X, y = ReadIris()
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)

ada.fit(X_std, y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1,len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Sum-squared-error")

plt.show()