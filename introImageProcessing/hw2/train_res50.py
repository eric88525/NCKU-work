import glob
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import copy

train_config = {
    "model_name": "res-v1",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epoch": 10,
    "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "momentum": 0.9,
    "weight_decay": 1e-4
}

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

def get_train_val_test_data(batch_size=4 , only_test_dataset = False):  # dataloader

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # mean and std for ImageNet
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    
    # a set of common trasnformation combination for training
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # transformations for testing do not need to do fancy tricks
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    cat_samples  =  [ (p,0) for p in glob.glob("PetImages/Cat/*.jpg") ]
    dog_samples = [ (p,1) for p in glob.glob("PetImages/Dog/*.jpg")] 
        
    assert len(cat_samples) == len(dog_samples)
    n = len(cat_samples)
    
    print(f"cat: {len(cat_samples)} , dogs: {len(dog_samples)}")

    if only_test_dataset:
        test_samples = cat_samples[ int (n*0.9):] + dog_samples[ int (n*0.9):]
        test_set = cd_Dataset(test_samples , test_transform)
        return test_set

    train_samples = cat_samples[:int(n*0.8)] +  dog_samples[:int(n*0.8)]
    val_samples =  cat_samples[int(n*0.8):int(n*0.9)] + dog_samples[int(n*0.8):int(n*0.9)]
    test_samples = cat_samples[ int (n*0.9):] + dog_samples[ int (n*0.9):]


    print(f"train: {len(train_samples)} val: {len(val_samples)} test: {len(test_samples)}")


    train_set = cd_Dataset(train_samples , train_transform)
    val_set = cd_Dataset(val_samples , test_transform)
    test_set = cd_Dataset(test_samples , test_transform)


    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_set , batch_size = batch_size , shuffle= False)
    test_loader = DataLoader(test_set, batch_size=batch_size , shuffle=False)
    
    return train_loader,val_loader ,test_loader


class Res50_CD(nn.Module):  # classifier model

    def __init__(self):
        super(Res50_CD, self).__init__()

        self.classifier = nn.Sequential(
            models.resnet50(pretrained=True),
            nn.Linear(1000, 2)
        )

    def forward(self, inp):
        return self.classifier(inp)


def train(train_config, train_loader, eval_loader, test_loader=None):

    model = Res50_CD()
    model.train()
    model.to(train_config["device"])

    writer = SummaryWriter()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
        parameters, lr=train_config["learning_rate"], momentum=train_config["momentum"], weight_decay=train_config["weight_decay"])

    loss_func = nn.CrossEntropyLoss(reduction="mean")

    best_performance = -1
    best_model = None
    n_iter = 0

    for epoch_idx in range(train_config["epoch"]):

        model.train()
        epoch_loss = 0
        train_datacount = 0
        # check acc
        accuracy = 0
        epoch_iter = tqdm(
            train_loader, desc=f"epoch {epoch_idx} Loss: [{0:.3f}]")
        # loop over the training set
        for batch in epoch_iter:

            x, y = ( b.to(train_config["device"]) for b in batch )
            model_output = model(x)
            y = y.squeeze()
            batch_loss = loss_func(model_output, y)

            # update parameters
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()

            # check acc
            _, predicted = torch.max(model_output.data, dim=-1)

            train_datacount += x.shape[0]
            accuracy += (predicted == y).sum().item()

            writer.add_scalar('Loss/train',  epoch_loss / len(train_loader), n_iter)
            n_iter += 1

            epoch_iter.set_description(
                f"epoch {epoch_idx} Loss: [{batch_loss.item():.3f}]")

        writer.add_scalar('Acc/train',  accuracy / train_datacount, epoch_idx)

        eval_acc = test(model, eval_loader)
        writer.add_scalar("Acc/eval" , eval_acc , epoch_idx)

        if eval_acc > best_performance:
            best_performance = eval_acc
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(),
                           train_config["model_name"] + ".pt")
            print(f"save model as {train_config['model_name']}.pt")

    print("testing...")
    test_acc = test(model , test_loader)
    writer.add_scalar("Acc/test" , test_acc)

# given model & test_loader , return loss & acc
def test(model, test_loader):
    test_count = 0
    acc_count = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = (i.to(next(model.parameters()).device) for i in batch)
            _, pred = torch.max(model(x).data, 1)
            test_count += y.shape[0]
            acc_count += (pred == y.squeeze()).sum().item()

    return  acc_count / test_count

def main():
    train_loader, val_loader, test_loader = get_train_val_test_data(
        batch_size=train_config["batch_size"])
    train(train_config, train_loader, val_loader,test_loader)


if __name__ == "__main__":
    main()
