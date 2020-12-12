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


# %%
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# %%
#  train & validation dataset
train_loader, val_loader = LoadData(
    dataset=datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
).get_dataloader([0.8, 0.2])

#  test dataset
test_loader = LoadData(
    dataset=datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
).get_dataloader()


# %% tmp section
# from sklearn.datasets import load_boston

# train_loader, val_loader, test_loader = LoadData(
#     X_y=load_boston(return_X_y=True)
# ).get_dataloader([0.7, 0.2, 0.1])


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


# %% start from here!
if __name__ == "__main__":
    # setting
    model = Model()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    modelwrapper.train(train_loader, val_loader, max_epochs=5)
    # # resume training
    modelwrapper.train(train_loader, val_loader, max_epochs=20)

    # evaluate the model
    print(f"\ntest loss: {modelwrapper.validation(test_loader)}")
    modelwrapper.classification_evaluate(test_loader, classes)
