import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('data.csv')

# 分析用データから機種名を削除
df = df.drop(columns='機種名')

# numpyに変更
array = df.to_numpy()

# クラスタ数n=2～データ数でクラスター分析を行う
n_range = [i for i in range(2, len(df))]
sse = [] # クラスター内誤差の平方和を格納する
for n in n_range:
    kmeans = KMeans(n_clusters=n).fit(array)
    sse.append(kmeans.inertia_)

# エルボー図のプロット
plt.plot(n_range, sse)
plt.xlabel('クラスター数')
plt.ylabel('SSE')
plt.show()
