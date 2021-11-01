# Apriori Algorithm & FP-Tree

# 1. Run
```
g++ main.cpp -o main
./main
```
# 2. Output
 result.txt: 所有可能的排列組合規則

![](https://i.imgur.com/a0W1t4x.png)

+ rule.txt : 所有的frequent itemsets

![](https://i.imgur.com/B7jjkUy.png)




# 3.Dataset 

| name               | transations | total items |
| ------------------ | ----------- | ----------- |
| IBM2021 (今年資料) | 4114        | 930         |
| IBM5000 (去年資料) | 828         | 52          |
| Groceries_dataset  | 14963       | 167         |

Groceries_dataset 網址: https://www.kaggle.com/heeraldedhia/groceries-dataset

### Groceries_dataset 
+ 物品的support count 偏低(平均為232)，因此要找規則的話support不能設定太高

![](https://i.imgur.com/fE3fgbJ.png)


# 4. Run time

+ 可以看到在support低時，Apriori算法耗時會增加許多，而FP-Growth都很快速
+ 兩份資料集的support都不高，在support=0.05時大部分item都已經被去除

### IBM2021
#### confidence = 0.001

| support | [FP/Apriori] run time (ms)   | rules | frequent itemset |
| ------- | --------------------------- | ----- | ---------------- |
| 0.01    | 211  / 63475                | 17706 | 1530             |
| 0.015   | 91  / 110                   | 3924  | 516              |
| 0.05    | 15 / 32                     | 0     | 1                |
| 0.1     | 15 /32                      | 0     | 0                |


### Groceries_dataset

#### confidence = 0.001
| support | run time (ms)  [FP/Apriori] | rules | frequent itemset |
| ------- | --------------------------- | ----- | ---------------- |
| 0.01    | 122  / 4508                 | 10    | 69               |
| 0.015   | 102  / 2668                 | 0     | 50               |
| 0.05    | 48 / 190                    | 0     | 11               |
| 0.1     | 31 / 59                     | 0     | 3                |

可由此看出，此資料集物品間的關聯性不高，最高的confidence也只有0.12左右

# 5.Analyze
+ 本次助教提供的資料即屬於low support high confidence

![](https://i.imgur.com/H4tao1B.png)

+ kaggle data 屬於low support low confidence 

![](https://i.imgur.com/x9AvDC7.png)


