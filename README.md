# camera-kmeans
Categorizing mirrorless cameras based on k-means

# 概要
スペックシートにある情報をベースに，ミラーレスカメラを分類するコード．

# なかみ
- data.csv
    - 食わせるデータ
- main.py
    - data.csvをもとに分類するコード
- optimizeNumberOfClusters.py
    - 最適なクラスター数の検討用コード（エルボー法）

# 使い方
- optimizeNumberOfClusters.pyで図示されるグラフから，最適なクラスター数を検討する
- main.py中のn_clustersの値を検討した値に置き換えて実行する

# データ項目
- 機種名
    - 機種名（分類には用いない）
- センサーサイズ
    - 1：フルサイズ
    - 2：APS-C
    - 3：m4/3
- 有効画素（万画素）
- 最高ISO（標準）
- 液晶モニタ大きさ（インチ）
- 液晶モニタドット（万）
- 液晶モニタ（タイプ）
    - 1：固定
    - 2：チルト可動式
    - 3：バリアングル
- ファインダー倍率
- 防塵防滴
    - 1：あり
    - 0：なし
- タッチパネル
    - 1：あり
    - 0：なし
- 手ブレ補正
    - 1：あり
    - 0：なし
- 位相差AF検出点
- コントラストAF検出点
- 連写（コマ/秒）
- 重量（g）
