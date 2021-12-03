import torch
import matplotlib.pyplot as plt
from imageseg import config
from imageseg.dataset import SegDataset
from torchvision import transforms

train_img = config.IMG_PATH[:10]
train_label = config.LABEL_PATH[:10]

img_transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                        config.INPUT_IMAGE_WIDTH)),
                                     transforms.ToTensor()])
trainDS = SegDataset(img_path=train_img, label_path=train_label,
                     transforms=img_transforms)
