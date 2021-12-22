# USAGE
# python train.py
# import the necessary packages
from imageseg import dataset
from imageseg.dataset import SegDataset
from imageseg.model import UNet
from imageseg import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os


def get_train_test_loader():
    
    trainLoader, testLoader = None, None

    img_transforms = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                            config.INPUT_IMAGE_WIDTH)),
                                         transforms.ToTensor()])

    if config.TEST_SPLIT > 0:
        split = train_test_split(config.IMG_PATH, config.LABEL_PATH,
                                 test_size=config.TEST_SPLIT, random_state=42)

        # unpack the data split
        train_img, test_img = split[:2]

        train_label, test_label = split[2:]

        print(f"Train data: {len(train_img)} Test data: {len(test_img)}")

        f = open(config.TEST_PATHS, "w")
        f.write("\n".join(test_img))
        f.close()

        testDS = SegDataset(img_path=test_img, label_path=test_label,
                            img_transforms=img_transforms)
        testLoader = DataLoader(testDS, shuffle=False,
                                batch_size=config.BATCH_SIZE,
                                num_workers=4)
    else:
        train_img = config.IMG_PATH
        train_label = config.LABEL_PATH

    trainDS = SegDataset(img_path=train_img, label_path=train_label,
                         img_transforms=img_transforms)

    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=config.BATCH_SIZE,
                             num_workers=4)

    return trainLoader, testLoader


def train(train_loader, test_loader=None):

    # model , loss , optimizer
    model = UNet(nbClasses=config.NUM_CLASSES).to(config.DEVICE)
    lossFunc = BCEWithLogitsLoss(reduction='mean')
    opt = Adam(model.parameters(), lr=config.INIT_LR)

    # record train&test loss
    H = {"train_loss": [], "test_loss": []}

    # start trainning
    print("[INFO] training the network...")

    # save best model by train_epoch_loss or test_epoch_loss
    best_performance = 1000

    startTime = time.time()
    for e in range(config.NUM_EPOCHS):

        model.train()
        epoch_train_loss = 0

        # loop over the training set
        for batch in tqdm(train_loader):

            (x, y) = (batch[0].to(config.DEVICE), batch[1].to(config.DEVICE))

            # caculate loss
            pred = model(x)
            loss = lossFunc(pred, y)

            # update parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        H["train_loss"].append(epoch_train_loss)

        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print(f"Train loss: {epoch_train_loss:.6f}")

        if test_loader != None:
            epoch_test_loss = test(model, test_loader)
            H["test_loss"].append(epoch_test_loss)
            print(f"Test loss: {epoch_test_loss:.6f}")

            if epoch_test_loss < best_performance:
                best_performance = epoch_test_loss
                best_model = copy.deepcopy(model)

        else:
            if epoch_train_loss < best_performance:
                best_performance = epoch_train_loss
                best_model = copy.deepcopy(model)

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    torch.save(best_model, config.MODEL_PATH)
    print(f"save model as {config.MODEL_PATH}.pt")

    return H


def test(model, test_loader):

    lossFunc = BCEWithLogitsLoss(reduction='mean')
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
            pred = model(x)
            test_loss += lossFunc(pred, y).item()

    test_loss /= len(test_loader)

    return test_loss


def plot_loss(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)


if __name__ == "__main__":
    train_loader, test_loader = get_train_test_loader()
    H = train(train_loader, test_loader)
    plot_loss(H)
