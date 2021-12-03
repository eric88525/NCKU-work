from numpy.core.fromnumeric import shape
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torch


class SegDataset(Dataset):

    def __init__(self, img_path, label_path, img_transforms) -> None:
        self.img_path = img_path
        self.label_path = label_path
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, id):

        img_path = self.img_path[id]
        label_path = self.label_path[id]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.img_transforms is not None:
            image = self.img_transforms(image)
        # image now is 3,width,height

        label = cv2.imread(label_path, 0)
        label = self.img_transforms(label).squeeze()

        # divide to 3 class Black , Red , Green
        # (channel , width , height)
        class_label = torch.zeros(image.shape)

        # if convert to gray scale , black=0 , red = 38 , green = 75,
        class_label[0][label == 0] = 1.0
        class_label[1][label == 38] = 1.0
        class_label[2][label == 75] = 1.0

        return (image, class_label)
