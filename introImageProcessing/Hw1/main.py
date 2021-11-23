from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np

# the main page


class mainPage(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(mainPage, self).__init__()
        uic.loadUi('main.ui', self)

        self.load_template_img()

        # Part 1
        self.LoadImageButton.clicked.connect(self.load_img)
        self.ColorSeperationButton.clicked.connect(self.color_seperation)
        self.ColorTransButton.clicked.connect(self.color_transformation)
        self.BlendButton.clicked.connect(self.blending)

        # Part 2
        self.GaussianBlurButton.clicked.connect(
            lambda: self.smoothing(smooth_type="gaussian"))
        self.BilateralButton.clicked.connect(
            lambda: self.smoothing(smooth_type="bilateral"))
        self.MedianButton.clicked.connect(
            lambda: self.smoothing(smooth_type="median"))

        # Part 3

        # show image
        self.show()

    # load the image that need to be process
    def load_template_img(self):
        # image file
        self.sun_img = cv2.imread("Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")
        self.dog_strong = cv2.imread(
            "Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
        self.dog_weak = cv2.imread(
            "Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")
        self.lenna_white_noice_img = cv2.imread(
            "Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg")
        self.lenna_pepper_salt_img = cv2.imread(
            "Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg")
        self.house_img = cv2.imread("Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")
        self.square_img = cv2.imread(
            "Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png")

    # HW 1-1
    def load_img(self):
        img = self.sun_img.copy()
        cv2.imshow("HW 1-1", img)
        print(f"Height: {img.shape[0]}\nWidth: {img.shape[1]}")

    # HW 1-2
    def color_seperation(self):

        B_img, G_img, R_img = cv2.split(self.sun_img)

        zeros = np.zeros(self.sun_img.shape[:2], dtype="uint8")

        R_img = cv2.merge([zeros, zeros, R_img])
        G_img = cv2.merge([zeros, G_img, zeros])
        B_img = cv2.merge([B_img, zeros, zeros])

        cv2.imshow("HW 1-2 R", R_img)
        cv2.moveWindow("HW 1-2 R", self.geometry().x() -
                       R_img.shape[1], self.geometry().y())

        cv2.imshow("HW 1-2 G", G_img)
        cv2.moveWindow("HW 1-2 G", self.geometry().x(), self.geometry().y())

        cv2.imshow("HW 1-2 B", B_img)
        cv2.moveWindow("HW 1-2 B", self.geometry().x() +
                       B_img.shape[1], self.geometry().y())

    # HW 1-3
    def color_transformation(self):

        img = self.sun_img.copy()

        # show COLOR_BGR2GRAY
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("HW 1-3 (1)", gray_img)
        cv2.moveWindow("HW 1-3 (1)", self.geometry().x() -
                       int(gray_img.shape[1]/2), self.geometry().y())

        # show avg gray
        avg_gray = img.mean(axis=2)
        avg_gray_img = np.zeros(img.shape, dtype=np.uint8)
        avg_gray_img[:, :, 0] = avg_gray
        avg_gray_img[:, :, 1] = avg_gray
        avg_gray_img[:, :, 2] = avg_gray

        cv2.imshow("HW 1-3(2)", avg_gray_img)
        cv2.moveWindow("HW 1-3(2)", self.geometry().x() +
                       int(avg_gray_img.shape[1]/2), self.geometry().y())
    # HW 1-4

    def blending(self):
        strong_img = self.dog_strong.copy()
        weak_img = self.dog_weak.copy()
        blend_img = cv2.addWeighted(strong_img, 0, weak_img, 1.0, 0.0)

        def blend_function(v):
            alpha = v/255
            beta = (1.0 - alpha)
            blend_img = cv2.addWeighted(strong_img, alpha, weak_img, beta, 0.0)
            cv2.imshow("HW 1-4", blend_img)

        cv2.imshow("HW 1-4", blend_img)
        cv2.createTrackbar("Blend", "HW 1-4", 0,
                           255, blend_function)

    def smoothing(self, smooth_type: str):

        if smooth_type == "gaussian":
            img = self.lenna_white_noice_img.copy()
            result = cv2.GaussianBlur(img, (5, 5), 0)
            win_name = "HW 2-1"

        elif smooth_type == "bilateral":
            img = self.lenna_white_noice_img.copy()
            result = cv2.bilateralFilter(img, 9, 90, 90)
            win_name = "HW 2-2"

        elif smooth_type == "median":

            img = self.lenna_pepper_salt_img.copy()
            result_3 = cv2.medianBlur(img, 3)
            result_5 = cv2.medianBlur(img, 5)
            win_name = "HW 2-3"

            # show 3x3
            cv2.imshow(win_name + " 3x3", result_3)
            cv2.moveWindow(win_name + " 3x3", self.geometry().x() +
                           img.shape[1], self.geometry().y())
            # show 5x5
            cv2.imshow(win_name + " 5x5", result_5)
            cv2.moveWindow(win_name + " 5x5", self.geometry().x() +
                           img.shape[1]*2, self.geometry().y())
            return
        else:
            print("what?")
            return

        # show ori image
        cv2.imshow("ori-img", img)
        cv2.moveWindow("ori-img", self.geometry().x(), self.geometry().y())

        if smooth_type == "median":
            return
        # show processed img
        cv2.imshow(win_name, result)
        cv2.moveWindow(win_name, self.geometry().x() +
                       img.shape[1], self.geometry().y())

    # Q3
    def blur(self, blur_type: str):


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = mainPage()
    app.exec_()
