import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# データ読み込み
df = pd.read_csv('data.csv')

# 機種名列抜き出し
nameDf = df['機種名']

# 分析用データから機種名を削除
df = df.drop(columns='機種名')

# numpyに変更
array = df.to_numpy()

# K-Means分析しndarrayをdataframeに変換
pred = KMeans(n_clusters=4).fit_predict(array)
predDf = pd.Series(pred)

# 機種名列とカテゴリを統合
resultDf = pd.DataFrame()
resultDf['機種名'] = nameDf
resultDf['カテゴリ'] = pred
resultDf = resultDf.sort_values('カテゴリ')
print(resultDf)