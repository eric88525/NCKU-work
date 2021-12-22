from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np
from numpy.lib.type_check import imag

# the main page


class mainPage(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(mainPage, self).__init__()
        uic.loadUi('main.ui', self)

        self.load_template_img()

        # Q1
        self.LoadImageButton.clicked.connect(self.load_img)
        self.ColorSeperationButton.clicked.connect(self.color_seperation)
        self.ColorTransButton.clicked.connect(self.color_transformation)
        self.BlendButton.clicked.connect(self.blending)

        # Q2
        self.GaussianBlurButton.clicked.connect(
            lambda: self.smoothing(smooth_type="gaussian"))
        self.BilateralButton.clicked.connect(
            lambda: self.smoothing(smooth_type="bilateral"))
        self.MedianButton.clicked.connect(
            lambda: self.smoothing(smooth_type="median"))

        # Q3
        self.EdgeGaussianBlurButton.clicked.connect(
            lambda: self.blur(blur_type="gaussian"))
        self.SobelXButton.clicked.connect(
            lambda: self.blur(blur_type="sobel_x"))
        self.SobelYButton.clicked.connect(
            lambda: self.blur(blur_type="sobel_y"))
        self.MagnitudeButton.clicked.connect(
            lambda: self.blur(blur_type="magnitude"))

        # Q4
        self.ResizeButton.clicked.connect(self.resize_img)
        self.TranslationButton.clicked.connect(self.transalation)
        self.RotationScalButton.clicked.connect(self.rotation_scale)
        self.ShearingButton.clicked.connect(self.shearing)

        # show
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
        self.gaussian_house = None
        self.resized_square_img = None

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
                       100, self.geometry().y())

        # show avg gray
        avg_gray = img.mean(axis=2)
        avg_gray_img = np.zeros(img.shape, dtype=np.uint8)
        avg_gray_img[:, :, 0] = avg_gray
        avg_gray_img[:, :, 1] = avg_gray
        avg_gray_img[:, :, 2] = avg_gray

        cv2.imshow("HW 1-3(2)", avg_gray_img)
        cv2.moveWindow("HW 1-3(2)", self.geometry().x() +
                       100, self.geometry().y())
    # HW 1-4

    def blending(self):
        strong_img = self.dog_strong.copy()
        weak_img = self.dog_weak.copy()
        blend_img = cv2.addWeighted(strong_img, 0, weak_img, 1.0, 0.0)

        # update whem trackbar change
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

        elif smooth_type == "bilateral":
            img = self.lenna_white_noice_img.copy()
            result = cv2.bilateralFilter(img, 9, 90, 90)

        elif smooth_type == "median":

            img = self.lenna_pepper_salt_img.copy()
            result_3 = cv2.medianBlur(img, 3)
            result_5 = cv2.medianBlur(img, 5)

            # show 3x3
            cv2.imshow(smooth_type + " 3x3", result_3)
            cv2.moveWindow(smooth_type + " 3x3", self.geometry().x() +
                           img.shape[1], self.geometry().y())
            # show 5x5
            cv2.imshow(smooth_type + " 5x5", result_5)
            cv2.moveWindow(smooth_type + " 5x5", self.geometry().x() +
                           img.shape[1]*2, self.geometry().y())

        else:
            print("what?")
            return
        # show ori image
        cv2.imshow("ori-img", img)
        cv2.moveWindow("ori-img", self.geometry().x(), self.geometry().y())

        if smooth_type == "median":
            return
        # show processed img
        cv2.imshow(smooth_type, result)
        cv2.moveWindow(smooth_type, self.geometry().x() +
                       img.shape[1], self.geometry().y())

    # Q3

    def blur(self, blur_type: str):

        kernel = None
        # create gaussian img once
        if type(self.gaussian_house) != np.array:
            gray_img = cv2.cvtColor(self.house_img, cv2.COLOR_BGR2GRAY)
            gaussian_kernel = np.array([[0.045, 0.122, 0.045], [0.122, 0.332, 0.122], [
                0.045, 0.122, 0.045]])
            self.gaussian_house = self.conv2D(gray_img, gaussian_kernel)

        if blur_type == "gaussian":
            result = self.gaussian_house

        elif blur_type == "sobel_x":
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            result = self.conv2D(self.gaussian_house, kernel)

        elif blur_type == "sobel_y":
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            result = self.conv2D(self.gaussian_house, kernel)

        elif blur_type == "magnitude":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            img_x = self.conv2D(self.gaussian_house,
                                kernel_x).astype("float32")
            img_y = self.conv2D(self.gaussian_house,
                                kernel_y).astype("float32")

            mag_img = np.sqrt(img_x * img_x + img_y*img_y)
            # norm
            mag_img = (mag_img - mag_img.min()) / \
                (mag_img.max() - mag_img.min())

            assert mag_img.shape == img_x.shape
            result = mag_img

        cv2.imshow(blur_type,  result)

    def conv2D(self, img, kernel):

        image_H, image_W = img.shape[:2]
        kernel_H, kernel_W = kernel.shape[:2]

        # create padded img
        pad = (kernel_W - 1) // 2
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # conv2d
        result = np.zeros([image_H, image_W], dtype=np.float32)
        for y in np.arange(pad, image_H + pad):
            for x in np.arange(pad, image_W + pad):
                # take kernel size if image
                roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                # sum of kernel value
                k = (roi * kernel).sum()
                # save caculate result
                result[y - pad, x - pad] = min(max(0, k), 255)

        result = result.astype("uint8")

        return result

    # Q4
    def resize_img(self, get_img=False):

        img = cv2.resize(self.square_img, [256, 256])

        if get_img:
            return img
        else:
            cv2.imshow("HW 4-1", img)
            cv2.moveWindow("HW 4-1", self.geometry().x(), self.geometry().y())

    def transalation(self,  get_img=False):

        img = self.resize_img(get_img=True)
        M = np.float32([[1, 0, 0], [0, 1, 60]])
        img = cv2.warpAffine(img, M, [400, 300])

        if get_img:
            return img
        else:
            cv2.imshow("HW 4-2", img)

    def rotation_scale(self, get_img=False):
        img = self.transalation(get_img=True)
        angle = 10
        scale = 0.5
        center = (128, 188)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, [400, 300])
        if get_img:
            return img
        else:
            cv2.imshow("HW 4-3", img)

    def shearing(self, get_img=False):
        img = self.rotation_scale(get_img=True)
        old_loc = np.array(
            [[50.0, 50], [200, 50], [50, 200]], dtype=np.float32)
        new_loc = np.array(
            [[10.0, 100], [200, 50], [100, 250]], dtype=np.float32)

        M = cv2.getAffineTransform(old_loc, new_loc)
        img = cv2.warpAffine(img, M, [400, 300])
        if get_img:
            return img
        else:
            cv2.imshow("HW 4-4", img)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = mainPage()
    app.exec_()
