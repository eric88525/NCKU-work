# AIMAS HW3

## Q56104076 陳哲緯

# install

```
pipenv shell
pipenv install
```

# fasttext

在本次作業中使用 fasttext 的預訓練模型 cc.zh.300.bin 來做為詞向量，f1 score 有所提升

# 枚舉參數

嘗試枚舉各種參數，找出最好的 F1 score
c1範圍 0.001 ~ 0.01，c2範圍0.001~0.01，共16種組合，最好F1成績為0.644
