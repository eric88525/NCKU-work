import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import re

# the list contain all img path
IMG_PATHS = glob.glob("./EKG_unzip/*/*.jpg")
IMG_PATHS = sorted( IMG_PATHS , key=lambda x:  int( str ( re.search("[0-9]*.jpg",x)[0][:-4]  )    )   )

# for check split image is correct
def show_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

# 12 part of img , return np.arr
def split_12_part(img):

    I   = img[ 365:513 , 122:424 ]
    II  = img[ 513:658 , 122:424 ]
    III = img[ 659:804 , 122:424 ]

    aVR = img[ 365:513 , 427:727 ]
    aVL = img[ 513:658 , 427:727 ]
    avF = img[ 659:804 , 427:727 ]

    V1 = img[ 365:513 , 733:1032 ]
    V2 = img[ 513:658 , 733:1032 ]
    V3 = img[ 659:804 , 733:1032 ]

    V4 = img[ 365:513 , 1037:1341 ]
    V5 = img[ 513:658 , 1037:1341 ]
    V6 = img[ 659:804 , 1037:1341 ]

    all = [ I , II , III , aVR , aVL , avF , V1 , V2 , V3 , V4 , V5 , V6 ]

    return all

# save array to pickle file
def save_to_pickle(data,save_path):

    with open(save_path,'wb') as f:
        pickle.dump( data , f)


def main():

    # result
    all_data = []

    print(f"There's {len(IMG_PATHS)} img need to be process")

    # split each img
    for img_path in tqdm(IMG_PATHS):

        img = cv2.imread(img_path)

        split_12 = split_12_part(img)

        all_data.append(split_12)

    assert len(IMG_PATHS) == len(all_data)

    # save to pickle file
    save_to_pickle(np.array(all_data , dtype=object) , "./Q1_data.pkl")

    print("save as ./Q1_data.pkl")

if __name__ == "__main__":
    main()