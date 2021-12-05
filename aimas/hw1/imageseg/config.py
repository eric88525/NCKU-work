import torch
import os

# base path of the dataset
BASE_PATH = "."

# all folder like 0_json , 1_json...
DATASET_PATH = os.listdir(os.path.join(BASE_PATH, "EKG_seg"))
DATASET_PATH = sorted(DATASET_PATH, key=lambda x: int(x[:-5]))
DATASET_PATH = [os.path.sep.join([BASE_PATH, "EKG_seg",  p])
                for p in DATASET_PATH]

# img path lile  ./EKG_seg/0_json/img.png
IMG_PATH = [os.path.sep.join(
    [p, "img.png"]) for p in DATASET_PATH]

# label path lile  ./EKG_seg/0_json/label.png
LABEL_PATH = [os.path.sep.join(
    [p, "label.png"]) for p in DATASET_PATH]

# define the test split
TEST_SPLIT = 0.1
# determine the device to be used for training and evaluation
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 2
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 4
# define the input image dimensions
INPUT_IMAGE_WIDTH = 486  # 橫向
INPUT_IMAGE_HEIGHT = 315  # 直向
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "Q3-model-test.pt")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "Q3-plot-test.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "Q3-test_paths.txt"])
