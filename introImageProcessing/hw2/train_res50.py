import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import torchvision.models as models
import cv2

from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import copy

train_config = {
    "model_name": "tt",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epoch": 6,
    "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "predtrained":False
}

class cd_Dataset(Dataset):
    def __init__(self , samples ,transform=None , mask=False ) -> None:
        self.samples = samples
        self.transform = transform
        self.mask = mask
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = cv2.imread(self.samples[index][0])

        if isinstance(img,type(None)): # error image
            img = cv2.imread(self.samples[0][0])

        if img.shape[-1] != 3:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mask:
            img = self.random_mask(img)

        img = self.transform(img)
        label = torch.Tensor([self.samples[index][-1]]).long()
        return  img , label

    def random_mask(self , img, p= 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
        if np.random.uniform(0, 1) > p:
            return img
        # the cv2 read file format is (H,W,3)
        H,W = img.shape[0] , img.shape[1]
        area = W*H
        se = np.random.uniform(sl,sh) * area
        re = np.random.uniform(r1,1/r1)
        he = int(np.sqrt(se*re))
        we = int(np.sqrt(se/re))
        xe = int(np.random.uniform(0,W))
        ye = int(np.random.uniform(0,H))
        img[ye:ye+he,xe:xe+we,:] = [125, 122,114]
        return img

def get_train_val_test_data(batch_size=4 , only_test_dataset = False , mask=False):  # dataloader

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # mean and std for ImageNet
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    # a set of common trasnformation combination for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # transformations for testing do not need to do fancy tricks
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
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


    train_set = cd_Dataset(train_samples , train_transform , mask=True)
    val_set = cd_Dataset(val_samples , test_transform ,  mask=False)
    test_set = cd_Dataset(test_samples , test_transform ,  mask=False)


    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_set , batch_size = batch_size , shuffle= False)
    test_loader = DataLoader(test_set, batch_size=batch_size , shuffle=False)

    return train_loader,val_loader ,test_loader


class Res50_CD(nn.Module):  # classifier model

    def __init__(self , pretrained=False):
        super(Res50_CD, self).__init__()

        self.classifier = nn.Sequential(
            models.resnet50(pretrained=pretrained),
            nn.Linear(1000, 2)
        )

    def forward(self, inp):
        return self.classifier(inp)


def train(train_config, train_loader, eval_loader, test_loader):

    model = Res50_CD(pretrained=train_config['predtrained'])
    model.train()
    model.to(train_config["device"])

    writer = SummaryWriter(log_dir = f"./runs/{train_config['model_name']}")

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

            writer.add_scalar('Loss/train', batch_loss.item(), n_iter)
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
    writer.add_scalar("Acc/test" , test_acc ,0)
    print("test acc " , test_acc)
    

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
    # pre
    train_config["model_name"] = "pretrained"
    train_config["predtrained"] = True
    train_loader, val_loader, test_loader = get_train_val_test_data(
        batch_size=train_config["batch_size"] , mask=False)
    train(train_config, train_loader, val_loader,test_loader)

    # no pre
    train_config["model_name"] = "no_pretrained"
    train_config["predtrained"] = False
    train_loader, val_loader, test_loader = get_train_val_test_data(
        batch_size=train_config["batch_size"] , mask=False)
    train(train_config, train_loader, val_loader,test_loader)

    # no pre + mask
    train_config["model_name"] = "mask"
    train_config["predtrained"] = False
    train_loader, val_loader, test_loader = get_train_val_test_data(
        batch_size=train_config["batch_size"] , mask=True)
    train(train_config, train_loader, val_loader,test_loader)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    main()
