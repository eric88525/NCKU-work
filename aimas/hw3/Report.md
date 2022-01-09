# AIMAS HW3

## Q56104076 陳哲緯

# install

```
pipenv shell
pipenv install
```
# 架構調整
+ 將原本的多個function整合並簡化，功能不變
```python
# 拿到 train data 和 test label
def get_embedding_label( data , fasttext_model): # given list of ('w',label) , return crf input x and label y
    x = [ [] for i in range(len(data))]
    y = [ [] for i in range(len(data))]

    for article_idx , article in enumerate(data):
        x[article_idx] =  [ {
                                **embedding_to_feature(  fasttext_model[word[0]] ) ,\
                            } for i , word in enumerate( article)]
        y[article_idx] = [ i[-1] for i in article]
    return x,y
```

# 資料清理
將部分全形字體轉半形
```
article_item_list = articles.replace('？','?').replace("…",'.').lower()
```


# fasttext

在本次作業中使用 fasttext 的預訓練模型 cc.zh.300.bin 來做為詞向量，預訓練語料為Common Crawl and Wikipedia，並且使用cbow架構，加入後f1 score 有所提升

```python
# fast text model:  "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz"
model = fasttext.load_model('./dataset/cc.zh.300.bin')
```

# 枚舉參數

嘗試枚舉各種參數，找出最好的 F1 score
c1範圍 0.001 ~ 0.01，c2範圍0.001~0.01，共16種組合

```python
for c1 in np.arange(0.001,0.01,0.003):
    for c2 in np.arange(0.001,0.01,0.003):
        y_pred, y_pred_mar, f1score , crf_model = train_test(x_train, y_train, x_test, y_test,c1,c2)
        if best_f1[0] < f1score:
            best_f1 = f1score , c1,c2            
        exp_result.append([c1, c2, f1score])
```

# iterration 調整
iter次數關係到訓練時間，嘗試調整為50、30個iter，可以看到iter在達到某個上限後，分數就不太會成長，從實驗結果觀察到在50個iter時能有比100更好的成績和更快的訓練速度

## fasttext as embedding 的實驗成果
+ 100 iter best: **0.628**
+ 50 iter best: **0.649**
+ 30 iter best: **0.637**

|c1|c2|100iter|50iter|30iter|
|-|-|-|-|-|
|0.001|0.001|0.585313|0.592419|0.605966|
|0.001|0.004|0.575615|0.635242|0.586633|
|0.001|0.007|0.588774|0.604795|0.576857|
|0.001|0.010|0.597280|==0.649579==|0.585787|
|0.004|0.001|0.608607|0.593548|0.582049|
|0.004|0.004|0.594039|0.632140|0.614845|
|0.004|0.007|0.606396|0.606176|0.591066|
|0.004|0.010|0.595893|0.630454|0.580986|
|0.007|0.001|==0.628510==|0.578430|0.586689|
|0.007|0.004|0.598557|0.599953|0.595671|
|0.007|0.007|0.605310|0.617775|0.593858|
|0.007|0.010|0.588083|0.584230|0.602389|
|0.010|0.001|0.598213|0.572475|0.601477|
|0.010|0.004|0.581723|0.619190|0.607663|
|0.010|0.007|0.603069|0.600434|0.603118|
|0.010|0.010|0.588515|0.599729|==0.637147==|



# 加入詞性標註
使用套件為jeiba，在每個字詞的300維度再增加詞性標註，像是名詞、動詞等..
分數相較於單純fasttext能有些許提升

```python
def get_pos_embedding_label( data , fasttext_model): 
    # given list of ('w',label) , return crf input x and label y
    x = [ [] for i in range(len(data))]
    y = [ [] for i in range(len(data))]

    for article_idx , article in enumerate(data):
        # 還原原本文章並做標記
        ori_articles = ''.join([i[0] for i in article])
        # jeiba pos tag
        pos_taggings = []
        for word,flag in  pseg.cut(ori_articles):
            pos_taggings += [flag] * len(word)
        # 加入到特徵內
        x[article_idx] =  [ {   
                                # add pos tag feature
                                **{'pos_tag': pos_taggings[i]},
                                # add embedding feature
                                **embedding_to_feature(  fasttext_model[word[0]] ) ,
                                              
                            } for i , word in enumerate( article)]
        y[article_idx] = [ i[-1] for i in article]

    return x,y
```


+ fasttext + 詞性，50iter，best: **0.651**

|c1|c2|f1-score|
|-|-|-|
|0.001|0.001|0.610164|
|0.001|0.004|0.611010|
|0.001|0.007|0.647875|
|0.001|0.010|0.637290|
|0.004|0.001|0.629820|
|0.004|0.004|0.620533|
|0.004|0.007|0.634148|
|0.004|0.010|0.621558|
|0.007|0.001|0.610002|
|0.007|0.004|0.646743|
|0.007|0.007|0.632605|
|0.007|0.010|0.633731|
|0.010|0.001|==0.651756==|
|0.010|0.004|0.639249|
|0.010|0.007|0.614353|
|0.010|0.010|0.636741|

# 心得
CRF能比神經網路能更加快速完成訓練，且能學到B-I-0的規則，所以常看到在LSTM或BERT後面加上CRF來限制輸出。在以往AI-CUP比賽中有實作過BERT+CRF的程式，輸出相較於直接預測每個TOKEN的分類，減少了更多B-I-O順序的違規。
