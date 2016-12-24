import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from AdalineSGD import AdalineSGD
from sklearn.linear_model import LogisticRegression

plt.style.use('ggplot')

def make_data(N, draw_plot=True, is_confused=False, confuse_bin=50):
    '''N個のデータセットを生成する関数
    データをわざと複雑にするための機能 is_confusedを実装する
    '''
    np.random.seed(1) # シードを固定して、乱数が毎回同じ出力になるようにする

    feature = np.random.randn(N, 2)
    df = pd.DataFrame(feature, columns=['x', 'y'])

    # 2値分類の付与：人為的な分離線の上下どちらに居るかで機械的に判定
    df['c'] = df.apply(lambda row : 1 if (5*row.x + 3*row.y - 1)>0 else 0,  axis=1)

    # 撹乱:データを少し複雑にするための操作
    if is_confused:
        def get_model_confused(data):
            c = 1 if (data.name % confuse_bin) == 0 else data.c
            return c

        df['c'] = df.apply(get_model_confused, axis=1)

    # 可視化：どんな感じのデータになったか可視化するモジュール
    # c = df.c つまり2値の0と1で色を分けて表示するようにしてある
    if draw_plot:
        plt.scatter(x=df.x, y=df.y, c=df.c, alpha=0.6)
        plt.xlim([df.x.min() -0.1, df.x.max() +0.1])
        plt.ylim([df.y.min() -0.1, df.y.max() +0.1])

    return df


def draw_split_line(weight_vector):
    a,b,c = weight_vector
    x = np.array(range(-10,10,1))
    y = (a * x + c)/-b
    plt.plot(x,y, alpha=0.3)

df = make_data(1000)
df.head(5)
# plt.show()

X = np.c_[ df['x'], df['y']]
y = df['c']

lr = AdalineSGD(eta=0.5,n_iter=100)

lr.fit(X,y)

# plot_decision_regions(X, y, classifier=lr)
print(lr.w_)

draw_split_line(lr.w_)


plt.show()

plt.plot(range(1,len(lr.cost_)+1), lr.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel("Average Cost")

plt.show()
