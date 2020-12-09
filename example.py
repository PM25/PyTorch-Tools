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
gpu = True if torch.cuda.is_available() else False

if gpu:
    print(f"*Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
else:
    print("*Using CPU")
    device = torch.device("cpu")


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
    trainset, batch_size=128, shuffle=True, num_workers=0
)

#  test dataset
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0
)


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
def validation(device, model, val_loader):
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
def evaluate(device, model, test_loader):
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


# %% training
def train(
    device, model, train_loader, epochs, loss_func, optimizer, multi_gpus=True, log=100
):
    # model setup
    model.train().to(device)
    if multi_gpus and torch.cuda.device_count() > 1:
        print(f"*Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # early stopping instance
    early_stopping = EarlyStopping(patience=5)

    # training start!
    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for step, data in enumerate(train_loader, start=1):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

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
                    f"[{epoch}/{epochs}, {step}/{len(train_loader)}] loss: {running_loss / step :.3f}"
                )

        # train & validation loss
        train_loss = running_loss / len(train_loader)
        val_loss = validation(device, model, val_loader)
        print(f"train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

        early_stopping(model, val_loss, optimizer)
        if early_stopping.early_stop:
            print("*Early Stopping.")
            break

    print("*Finished Training!")
    return early_stopping.checkpoint


# %% start from here!
if __name__ == "__main__":
    # init model
    model = Model()
    # setting
    epochs = 30
    loss_func = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training result (use checkpoint class to load best model)
    checkpoint = train(device, model, train_loader, epochs, loss_func, optimizer)

    null_model = Model().to(device)
    null_optimizer = optim.Adam(null_model.parameters(), lr=lr)
    checkpoint_data = checkpoint.load(null_model, null_optimizer)

    # evaluate the model
    model = checkpoint_data["model"]
    evaluate(device, model, test_loader)

# %%
