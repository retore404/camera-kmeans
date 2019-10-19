import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("data.csv")
df = df.drop(columns="機種名")

array = df.to_numpy()


pred = KMeans(n_clusters=4).fit_predict(array)
print(pred)