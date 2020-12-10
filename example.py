# %%
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.utils.prune as prune
from torch.utils.data import random_split


# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# %%
#  train & validation dataset
dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainset_count = int(len(dataset) * 0.8)
valset_count = len(dataset) - trainset_count
trainset, valset = random_split(dataset, [trainset_count, valset_count])

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=True, num_workers=0
)

#  test dataset
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0
)


#%%
def get_default_device(verbose=False):
    gpu = True if torch.cuda.is_available() else False

    if gpu:
        if verbose:
            print(f"*Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        if verbose:
            print("*Using CPU")
        device = torch.device("cpu")

    return device


#%%
def get_default_optimizer(model, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


# %%
#  Cifar-10's classes
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# model = models.resnet50(pretrained=True)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %% validation
def validation(model, val_loader, device=None):
    if device is None:
        device = get_default_device()
    model.eval().to(device)
    
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(val_loader)


# %% evaluate
def evaluate(model, test_loader, device=None):
    if device is None:
        device = get_default_device()
    model.eval().to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the {len(test_loader)} test inputs: {(100 * correct / total)} %"
    )

    class_correct = list(0.0 for i in range(len(classes)))
    class_total = list(0.0 for i in range(len(classes)))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print(
            f"Accuracy of {classes[i]: >5} : {100 * class_correct[i] / class_total[i]:.0f} %"
        )


# %%
class TrainingSetup:
    def __init__(self, device=None, max_epochs=1000, multi_gpus=True, early_stopping=True, log=100):
        self.max_epochs = max_epochs
        self.multi_gpus = multi_gpus
        self.early_stopping = early_stopping
        self.log = log
        if device is None:
            self.device = get_default_device(verbose=True)
        else:
            self.device = device

    # train model
    def train(self, model, train_loader, loss_func, optimizer=None):
        if optimizer is None:
            optimizer = get_default_optimizer(model)
        # model setup
        model.train().to(device)
        if self.multi_gpus and torch.cuda.device_count() > 1:
            print(f"*Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        # early stopping instance
        if(self.early_stopping):
            early_stopping = EarlyStopping(patience=5)

        # training start!
        for epoch in range(1, max_epochs + 1):
            running_loss = 0.0

            for step, data in enumerate(train_loader, start=1):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

                if step % 100 == 0 or step == len(train_loader):
                    print(
                        f"[{epoch}/{max_epochs}, {step}/{len(train_loader)}] loss: {running_loss / step :.3f}"
                    )

            # train & validation loss
            train_loss = running_loss / len(train_loader)
            val_loss = validation(model, val_loader, self.device)
            print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

            if(self.early_stopping):
                early_stopping(model, val_loss, optimizer)
                if early_stopping.get_early_stop() == True:
                    print("*Early Stopping.")
                    break

        print("*Finished Training!")
        if(self.early_stopping):
            checkpoint = early_stopping.get_checkpoint()
        else:
            checkpoint = Checkpoint()
            checkpoint.tmp_save(model, optimizer, epoch, val_loss)
        return checkpoint
        


# %% start from here!
if __name__ == "__main__":
    # init model
    model = Model()
    # setting
    loss_func = nn.CrossEntropyLoss()
    device = get_default_device(verbose=True)
    max_epochs = 5
    train_setup = TrainingSetup(device, max_epochs, early_stopping=False)

    # training result (use checkpoint class to load best model)
    checkpoint = train_setup.train(model, train_loader, loss_func)

    null_model = Model()
    null_optimizer = get_default_optimizer(null_model)
    checkpoint_data = checkpoint.load(null_model, null_optimizer)

    # evaluate the model
    model = checkpoint_data["model"]
    evaluate(model, test_loader)

# %%
