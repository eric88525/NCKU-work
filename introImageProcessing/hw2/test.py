import torch
from torch.utils.data import Dataset , DataLoader
import cv2
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
import numpy as np

class cd_Dataset(Dataset):
    def __init__(self , samples ,transform=None  ) -> None:
        
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img = Image.open(self.samples[index][0])

        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.transform(img)
        label = torch.Tensor([self.samples[index][-1]]).long()
        return  img , label


def check_img(samples):
    for sample in samples:
        i = Image.open( sample[0] )

def get_train_val_test_data(batch_size=4):  # dataloader

    cat_samples  =  [ (p,0) for p in glob.glob("PetImages/Cat/*.jpg") ]
    dog_samples = [ (p,1) for p in glob.glob("PetImages/Dog/*.jpg")] 

    check_img(cat_samples)
    check_img(dog_samples)
    assert len(cat_samples) == len(dog_samples)
    n = len(cat_samples)

    print(f"cat: {len(cat_samples)} , dogs: {len(dog_samples)}")

    train_samples = cat_samples[:int(n*0.8)] +  dog_samples[:int(n*0.8)]
    val_samples =  cat_samples[int(n*0.8):int(n*0.9)] + dog_samples[int(n*0.8):int(n*0.9)]
    test_samples = cat_samples[ int (n*0.9):] + dog_samples[ int (n*0.9):]


    print(f"train: {len(train_samples)} val: {len(val_samples)} test: {len(test_samples)}")

    #MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # mean and std for ImageNet
    #normalize = transforms.Normalize(mean=MEAN, std=STD)

    # a set of common trasnformation combination for training
    train_transform = transforms.Compose([
        transforms.Resize((400,400)),
     #   transforms.RandomResizedCrop(224),
     #   transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
     #   normalize
    ])

    # transformations for testing do not need to do fancy tricks
    test_transform = transforms.Compose([
        transforms.Resize((400,400)),
      #  transforms.Resize(256),
      #  transforms.CenterCrop(224),
        transforms.ToTensor(),
      #  normalize
    ])

    train_set = cd_Dataset(train_samples , train_transform)
    val_set = cd_Dataset(val_samples , test_transform)
    test_set = cd_Dataset(test_samples , test_transform)


    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_set , batch_size = batch_size , shuffle= False)
    test_loader = DataLoader(test_set, batch_size=batch_size , shuffle=False)
    
    return train_loader,val_loader ,test_loader


#train_loader, val_loader, test_loader = get_train_val_test_data()


