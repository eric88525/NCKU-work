# USAGE
# python predict.py
# import the necessary packages
from imageseg import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    plt.savefig("hi.jpg")


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(
            image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        groundTruthPath = imagePath.replace("img.png", "label.png")
        print("G TRUST = ", groundTruthPath)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH,
                                     config.INPUT_IMAGE_HEIGHT))
        gtMask = cv2.cvtColor(gtMask, cv2.COLOR_BGR2RGB)
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2,
                                     0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array

        predMask = model(image).squeeze()  # N,C,W,H -> C,W,H

        predMask = torch.sigmoid(predMask)
        predMask = torch.argmax(predMask, dim=0)
        predMask = predMask.cpu().numpy()

        color_mask = np.zeros((predMask.shape[0], predMask.shape[1], 3))
        color_mask[predMask == 1, 0] = 255
        color_mask[predMask == 2, 0] = 127
        color_mask[predMask == 2, 1] = 255

        print(color_mask.shape)

        # prepare a plot for visualization
        prepare_plot(orig, gtMask, color_mask)


# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    make_predictions(unet, path)
    break
