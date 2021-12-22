from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from train_res50 import *
from torchsummary import summary
import torchvision.models as models
import pickle


class Q5(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(Q5, self).__init__()

        uic.loadUi('Q5.ui', self)

        # load model & record
        self.load_model_and_record(
            record_path="loss_vgg-epo30.pkl", model_path="vgg-epo30.pt")

        # class name of dataset
        self.img_class_name = [
            'airplane', 'autombile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

        # buttons
        self.ShowTrainImageButton.clicked.connect(self.show_train_img)
        self.ShowHyperParameterButton.clicked.connect(
            self.show_hyper_parameter)
        self.ShowModelShortCutButton.clicked.connect(self.show_model_shortcut)
        self.ShowAccButton.clicked.connect(self.show_acc_button)
        self.TestButton.clicked.connect(self.test_model)

        self.show()

    def load_model_and_record(self, record_path="loss_vgg-epo30.pkl", model_path="vgg-epo30.pt"):

        # read train/test record
        record = None
        with open(record_path, "rb") as f:
            record = pickle.load(f)

        self.train_acc = [x['train_acc'] for x in record]
        self.test_acc = [x['test_acc'] for x in record]
        self.train_loss = [x['train_loss'] for x in record]

        # load model
        self.model = VGG_class10()
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def show_train_img(self):

        fig, _ = plt.subplots()
        fig.subplots_adjust(hspace=.3, wspace=.3)

        for i, (image, label) in enumerate(train_data, start=1):
            plt.subplot(3, 3, i)
            plt.axis('off')
            plt.title(self.img_class_name[label])
            plt.imshow(np.moveaxis(image.numpy(), 0, -1))
            if i == 9:
                break
        plt.show()

    def show_hyper_parameter(self):

        print(f"batch size: {train_config['batch_size']}")
        print(f"learning rate: {train_config['learning_rate']}")
        print(f"SGD")

    def show_model_shortcut(self):
        summary(models.vgg16(), (3, 32, 32))

    def show_acc_button(self):

        # ACC plot
        plt.subplot(2, 1, 1)
        x = [i for i in range(len(self.train_acc))]
        plt.plot(x, self.train_acc, label="train_acc")
        plt.plot(x, self.test_acc, label="test_acc")

        plt.xlabel("epoch")
        plt.ylabel("%")
        plt.legend(loc="lower right")

        # loss plot
        plt.subplot(2, 1, 2)
        x = [i for i in range(len(self.train_loss))]
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, self.train_loss, label="train_loss")

        plt.suptitle("Accuracy")

        plt.show()

    def test_model(self):

        idx = self.TestIndexInput.value()-1

        if idx >= len(test_data):
            print("out of idx")
            return

        img, _ = test_data[idx]

        plt.figure(figsize=(18, 5))

        # show img
        plt.subplot(1, 2, 1)
        plt.title("Test Image")
        plt.axis("off")
        temp_img_ = img.transpose(0, 1).transpose(1, 2)
        plt.imshow(temp_img_)

        # show pred
        with torch.no_grad():
            pred = torch.softmax(self.model(
                img.unsqueeze(0)), dim=-1).squeeze().numpy()
            x_ = [self.img_class_name[i] for i in range(10)]

            plt.subplot(1, 2, 2)
            plt.bar(x_,  pred)
            plt.show()


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = Q5()
    app.exec_()
