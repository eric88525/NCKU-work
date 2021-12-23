from PIL import Image
from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from train_res50 import get_train_val_test_data , Res50_CD
from torchsummary import summary
import torchvision.models as models
import pickle

test_dataset = get_train_val_test_data(only_test_dataset=True)


class Q5(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(Q5, self).__init__()

        uic.loadUi('Q5.ui', self)

        # load model & record
        self.load_model_and_record(
            record_path="tensorboard_record.png", model_path="res-v1.pt")

        # buttons
        self.showModelStructureButton.clicked.connect(self.showModelStructure)
        self.showTensorboardButton.clicked.connect(
            self.showTensorboard)
        self.TestButton.clicked.connect(self.Test)
        self.DataArgumentButton.clicked.connect(self.DataArgument)

        self.show()

    def load_model_and_record(self, record_path="loss_vgg-epo30.pkl", model_path="vgg-epo30.pt"):

        # read train/test record
        self.tensorboard_img = cv2.imread(record_path)
        # load model
        self.model = Res50_CD()
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def showModelStructure(self):
        summary(self.model, (3, 32, 32))

    def showTensorboard(self):
        cv2.imshow("Q5-2" , self.tensorboard_img)

    def Test(self):
        idx = self.imgIndex.value()-1
        with torch.no_grad():
            img = cv2.imread(test_dataset.samples[idx][0])
            inp ,_ = test_dataset[idx]
            inp = inp.unsqueeze(0)
            print(inp.shape)
            try:
                pred = torch.argmax(self.model( inp ) , dim = -1)
                print(pred)
                if pred.item() == 0:
                    cv2.imshow(  "cat",img )
                else:
                    cv2.imshow(  "dog",img )
            except:
                print(f"Image Error! {test_dataset.samples[idx]}")
                pass


    def DataArgument(self):
        img = Image.open("PetImages/Cat/2.jpg")
        img.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Q5()
    app.exec_()

