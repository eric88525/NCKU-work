import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


random.seed(2021)
np.random.seed(2021)


# 14 parameters
attributes = ["brand", "RGB-keyboard", "screen size", "weight", "cpu", "ram", "disk", "graph card", "wide color", "monitor calibration", "cooling boost", "fresh rate", "cpu boost",
              'body color']

describe = [["ROG-品牌", "MSI-品牌", "HP-品牌", "ASUS-品牌", "VAIO-品牌"],
            ['有-RGB鍵盤背光', '無-RGB鍵盤背光'],
            ['13-螢幕尺寸', '15-螢幕尺寸', '17-螢幕尺寸'],
            ['1-2 kg-重量', '2kg up-重量'],
            ['i5-處理器', 'i7-處理器'],
            ['8-記憶體大小', '16-記憶體大小', '32-記憶體大小'],
            ['128-硬碟大小', '512-硬碟大小', '1T-硬碟大小'],
            ['無-顯示卡', 'RTX-顯示卡', 'MX-顯示卡'],
            ['有-廣色域覆蓋', '沒-廣色域覆蓋'],
            ['有-校色器', '沒-校色器'],
            ['有-散熱提升', '沒-散熱提升'],
            ['60HZ-刷新率', '120HZ UP-刷新率'],
            ['有-超頻功能', '沒-超頻功能'],
            ['red', 'white', 'black', 'other']]

# laptop kind
gaming_laptop = [[0.25, 0.25, 0.25, 0.25, 0], [0.9, 0.1], [0.1, 0.7, 0.2], [0.2, 0.8], [0.3, 0.7], [
    0.1, 0.7, 0.2], [0, 0.8, 0.2], [0, 1, 0], [0.1, 0.9], [0, 1], [0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.25, 0.25, 0.25, 0.25]]
business_laptop = [[0, 0.25, 0.25, 0.25, 0.25], [0.1, 0.9], [0.8, 0.1, 0.1], [0.95, 0.05], [0.8, 0.2], [
    0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.45, 0.1, 0.45], [0.2, 0.8], [0, 1], [0.1, 0.9], [0.95, 0.05], [0.1, 0.9], [0.25, 0.25, 0.25, 0.25]]
ceator_laptop = [[0, 0.34, 0.33, 0.33, 0], [0.3, 0.7], [0.1, 0.8, 0.1], [0.5, 0.5], [0.2, 0.8], [
    0.1, 0.8, 0.1], [0, 0.5, 0.5], [0.1, 0.9, 0], [0.8, 0.2], [0.3, 0.7], [0.8, 0.2], [0.95, 0.05], [0.1, 0.9], [0.25, 0.25, 0.25, 0.25]]


def create_sample(attribute_p):

    attr = []

    for i, attr_p in enumerate(attribute_p):
        col_attr = np.random.choice(
            [x for x in range(len(describe[i]))], p=attr_p)
        attr.append(col_attr)
    return attr


def main():

    X = []
    Y = []

    for i in range(1000):

        X.append(create_sample(gaming_laptop))
        Y.append("gaming_laptop")

        X.append(create_sample(business_laptop))
        Y.append("business_laptop")

        X.append(create_sample(ceator_laptop))
        Y.append("ceator_laptop")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = DecisionTreeClassifier(
        max_features=14, max_depth=5, min_samples_split=10, min_samples_leaf=20)
    model.fit(X_train, y_train)
    print('train model done')
    print(model.score(X_test, y_test))

    # visual
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=attributes,
                   class_names=model.classes_, filled=True, max_depth=5, fontsize=6)
    plt.savefig("tree.png", bbox_inches='tight', dpi=150)
    # plt.show()


if __name__ == "__main__":
    main()
