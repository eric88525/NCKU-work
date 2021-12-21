from PyQt5 import QtWidgets, uic
import sys
import cv2
import numpy as np

class mainPage(QtWidgets.QMainWindow, QtWidgets.QDialog):
    def __init__(self):
        super(mainPage, self).__init__()
        uic.loadUi('Q2.ui', self)
        # Q2 imgs
        self.imgs = [ cv2.imread(f"Dataset_OpenCvDl_Hw2/Q2_Image/{x}.bmp") for x in range(1,16)]

        self.findCornersButton.clicked.connect(self.findCorners)
        self.findIntrinsticButton.clicked.connect(self.findIntrinstic)
        self.findExtrinsicButton.clicked.connect(self.findExtrinsic)
        self.findDistrotionButton.clicked.connect(self.findDistrotion)
        self.showResultButton.clicked.connect(self.showResult)

        self.show()

    def findCorners(self):
        CHECKER_BOARD = (11,8)
        world_3d = np.zeros((1, CHECKER_BOARD[0] * CHECKER_BOARD[1], 3), np.float32)
        world_3d[0,:,:2] = np.mgrid[0:CHECKER_BOARD[0], 0:CHECKER_BOARD[1]].T.reshape(-1, 2)
        objpoints = [ world_3d for i in range(15) ]
        imgpoints = []

        for img in self.imgs:
            gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKER_BOARD, None)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, CHECKER_BOARD, corners, ret)
            cv2.imshow('2-1', cv2.resize(img, (800, 800)))
            cv2.waitKey(500)

        _, self.intrinsic_matrix, self.distortion_coefficients, self.rotation_vector, self.translation_vector = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        cv2.destroyAllWindows()

    def findIntrinstic(self):
        print("Intrinsic:")
        print(self.intrinsic_matrix)

    def findExtrinsic(self):
        img_idx = self.indexSelector.value()-1
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector[img_idx])
        ext = np.append(rotation_matrix, self.translation_vector[img_idx], axis=1)
        print("Extrinsic:")
        print(ext)

    def findDistrotion(self):
        print("Distortion:")
        print(self.distortion_coefficients)

    def showResult(self):
        for img in self.imgs:
            uds = cv2.undistort(img, self.intrinsic_matrix, self.distortion_coefficients)
            print(uds.shape)
            cv2.imshow("2-5 origin" , cv2.resize(img , (400,400)))
            cv2.moveWindow("2-5 origin", self.geometry().x() - 200, self.geometry().y())
            cv2.imshow("2-5 undistort",cv2.resize(uds , (400,400)) )
            cv2.moveWindow("2-5 undistort", self.geometry().x() + 200, self.geometry().y())
            cv2.waitKey(500)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainPage()
    app.exec_()