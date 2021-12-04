from imageseg import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def prepare_plot(origImage, origMask, predMask, idx):
    # initialize our figure
    figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
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
    plt.savefig(os.path.sep.join([config.BASE_OUTPUT, f"Q3-demo-{idx}.jpg"]))


def make_predictions(model, imagePath, idx):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():

        # origin image
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # model predict
        image = cv2.resize(
            image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        orig = image.copy()

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        predMask = model(image).squeeze()  # N,C,W,H -> C,W,H
        predMask = torch.sigmoid(predMask)  # C,W,H

        predMask = predMask.cpu().numpy()

        color_mask = np.zeros(
            (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH, 3))

        # green
        color_mask[(predMask[1] > 0.5), 1] = 254
        # red
        color_mask[(predMask[0] > 0.5), 0] = 254

        # label
        groundTruthPath = imagePath.replace("img.png", "label.png")
        gtMask = cv2.imread(groundTruthPath)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH,
                                     config.INPUT_IMAGE_HEIGHT))
        gtMask = cv2.cvtColor(gtMask, cv2.COLOR_BGR2RGB)

        # prepare a plot for visualization
        prepare_plot(orig, gtMask, color_mask, idx)


# load the image paths in our file and randomly select 10
print("[INFO] loading up test image paths...")
imagePaths = np.random.choice(config.IMG_PATH, size=10)

print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# predict and save to image

for i in range(10):
    print(f"Testing {imagePaths[i]}, save to Q3-demo-{i}.jpg")
    make_predictions(unet,  imagePaths[i], i)
