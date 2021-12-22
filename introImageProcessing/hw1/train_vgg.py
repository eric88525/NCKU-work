import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import copy

train_config = {
    "model_name": "vgg-epo512",
    "batch_size": 64,
    "learning_rate": 0.001,
    "epoch": 30,
    "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    "momentum": 0.9,
    "weight_decay": 1e-4
}


def get_train_test_data(batch_size=4):  # dataloader

    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()

    train_set = CIFAR10("Dataset", train=True,
                        transform=train_transform, download=True)
    test_set = CIFAR10("Dataset", train=False,
                       transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    return train_loader, test_loader


class VGG_class10(nn.Module):  # classifier model

    def __init__(self):
        super(VGG_class10, self).__init__()

        self.classifier = nn.Sequential(
            models.vgg16(),
            nn.Linear(1000, 10)
        )

    def forward(self, inp):
        return self.classifier(inp)


def train(train_config, train_loader, test_loader=None):

    model = VGG_class10()
    model.train()
    model.to(train_config["device"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.SGD(
        parameters, lr=train_config["learning_rate"], momentum=train_config["momentum"], weight_decay=train_config["weight_decay"])

    loss_func = nn.CrossEntropyLoss(reduction="mean")

    train_loss_record = [{} for i in range(train_config["epoch"])]

    if test_loader != None:
        best_performance = 0
    else:
        best_performance = 1000

    best_model = None

    for epoch_idx in range(train_config["epoch"]):

        model.train()
        epoch_loss = 0
        train_datacount = 0
        # check acc
        accuracy = 0

        for batch in tqdm(train_loader):

            x, y = batch
            x = x.to(train_config["device"])
            y = y.to(train_config["device"], dtype=torch.long)

            model_output = model(x)

            batch_loss = loss_func(model_output, y)

            # update parameters
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()

            # check acc
            _, predicted = torch.max(model_output.data, 1)
            train_datacount += len(x)
            accuracy += (predicted == y).sum().item()

        train_loss_record[epoch_idx]["train_loss"] = epoch_loss / \
            len(train_loader)

        print(
            f"epoch: {epoch_idx} epoch loss = {epoch_loss / len(train_loader) }")
        print(
            f"epoch: {epoch_idx} ACC = { accuracy /  train_datacount } ({accuracy}/{train_datacount})")

        train_loss_record[epoch_idx]["train_acc"] = accuracy / train_datacount
        # test part(acc)
        if test_loader != None:

            test_acc = test(model, test_loader)
            train_loss_record[epoch_idx]["test_acc"] = test_acc

            if test_acc > best_performance:
                best_performance = test_acc
                best_model = copy.deepcopy(model)

                torch.save(best_model.state_dict(),
                           train_config["model_name"] + ".pt")
                print(f"save model as {train_config['model_name']}.pt")
        else:
            if epoch_loss < best_performance:
                best_performance = epoch_loss
                best_model = copy.deepcopy(model)

                torch.save(best_model.state_dict(),
                           train_config["model_name"] + ".pt")
                print(f"save model as {train_config['model_name']}.pt")

    return train_loss_record


def test(model, test_loader):  # given model & test_loader , return acc

    test_count = 0
    acc_count = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch

            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)

            _, pred = torch.max(model(x).data, 1)

            test_count += y.shape[0]
            acc_count += (pred == y).sum().item()

    print(
        f"test end , acc rate: {acc_count / test_count :.2f} ({acc_count}/{test_count})")
    return acc_count / test_count


def save_to_pickle(data: object, fname):  # save obj to pickle
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def main():

    train_loader, test_loader = get_train_test_data(
        batch_size=train_config["batch_size"])

    train_loss_record = train(train_config, train_loader, test_loader)

    save_to_pickle(train_loss_record, f"loss_{train_config['model_name']}.pkl")


if __name__ == "__main__":
    main()
