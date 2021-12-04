import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
import glob
import re

# the list contain all img path
IMG_PATHS = glob.glob("./EKG_unzip/*/*.jpg")
IMG_PATHS = sorted(IMG_PATHS, key=lambda x:  int(
    str(re.search("[0-9]*.jpg", x)[0][:-4])))

# load from pickle file


def load_from_pickle(save_path):

    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data

# extract signal from image


def extract_signal(img):

    # convert BGR to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert gray img to black & white (signal is black)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # remove II mark
    thresh[9:24, 26:39] = 255

    # cv2.imwrite("q2.jpg",thresh)

    # convert to  signal
    signal = np.argmin(thresh, axis=0)

    # remove zero from begin and end
    start = 0
    end = signal.shape[0]-1

    while signal[start] == 0:
        start += 1

    while signal[end] == 0:
        end -= 1

    signal = signal[start:end+1]

    # flip signal
    signal = img.shape[0] - signal

    # moving avg 3
    smooth_signal = signal.copy()
    for i in range(1, signal.shape[0]-1):
        smooth_signal[i] = np.sum(signal[i-1:i+2])/3

    # fix value

    return smooth_signal


def detect_peaks(ecg_signal, threshold=0.45, qrs_filter=None):
    '''
    Peak detection algorithm using cross corrrelation and threshold 
    '''
    if qrs_filter is None:
        # create default qrs filter, which is just a part of the sine function
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    # normalize data
    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()

    # calculate cross correlation
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)

    # index of peak
    sim_index = np.where(similarity > threshold)[0]

    # mark peak as 1 and others as 0
    bin_signal = np.zeros(ecg_signal.shape)
    bin_signal[sim_index] = 1

    peaks = 0

    # find 0 to 1  moment (peak)
    for i in range(bin_signal.shape[0]-1):
        if (bin_signal[i+1] == 1 and bin_signal[i] == 0):
            peaks += 1

    return peaks


def main():

    # the data generate by Q1
    data = load_from_pickle("./Q1_data.pkl")

    # get last element(long lead II)
    all_long_lead_II = [d[-1] for d in data]

    result = []

    for img_idx, lead_II in enumerate(tqdm(all_long_lead_II)):

        # signal from img
        signal = extract_signal(lead_II)

        # peak index
        peaks = detect_peaks(signal)

        # 10s peaks * 6 = 10s peakss
        result.append([f"{IMG_PATHS[img_idx]}",  peaks*6])

    # save to csv
    df = pd.DataFrame(data=result, columns=["picture", "Rate"])
    df.to_csv("Q2.csv", index=False)


def debug():
    import re
    IMG_PATHS = glob.glob("./EKG_unzip/*/*.jpg")
    IMG_PATHS = sorted(IMG_PATHS, key=lambda x:  int(
        str(re.search("[0-9]*.jpg", x)[0][:-4])))
    print(IMG_PATHS[120:130])
    pass


if __name__ == "__main__":
    main()
    # debug()
