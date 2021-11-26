from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from train_vgg import get_train_test_data, train_config
from torchsummary import summary
import torchvision.models as models

# load dataset
train_data, test_data = get_train_test_data()
train_data = train_data.dataset
test_data = test_data.dataset

img_class_name = [
    'airplane', 'autombile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']


class Q5(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(Q5, self).__init__()
        uic.loadUi('Q5.ui', self)

        #
        self.ShowTrainImageButton.clicked.connect(self.show_train_img)
        self.ShowHyperParameterButton.clicked.connect(
            self.show_hyper_parameter)
        self.ShowModelShortCutButton.clicked.connect(self.show_model_shortcut)
        self.ShowAccButton.clicked.connect(self.show_acc_button)

        self.show()

    def show_train_img(self):

        fig, axs = plt.subplots()
        fig.subplots_adjust(hspace=.3, wspace=.3)

        for i, (image, label) in enumerate(train_data, start=1):
            plt.subplot(3, 3, i)
            plt.axis('off')
            plt.title(img_class_name[label])
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
        pass


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = Q5()
    app.exec_()
