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
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
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


# # %% tmp section
# from sklearn.datasets import load_boston

# data = load_boston()
# X, y = load_boston(return_X_y=True)

# X_train = torch.from_numpy(X).float()
# y_train = torch.from_numpy(y).float()
# dataset = torch.utils.data.TensorDataset(X_train, y_train)

# trainset_count = int(len(dataset) * 0.8)
# valset_count = len(dataset) - trainset_count
# trainset, valset = random_split(dataset, [trainset_count, valset_count])

# train_loader = torch.utils.data.DataLoader(
#     trainset, batch_size=32, shuffle=True, num_workers=0
# )

# val_loader = torch.utils.data.DataLoader(
#     valset, batch_size=32, shuffle=True, num_workers=0
# )

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
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    modelwrapper.train(train_loader, val_loader, max_epochs=50)

    # # resume training
    # modelwrapper.train(train_loader, val_loader, max_epochs=20)

    # evaluate the model
    # modelwrapper.classification_evaluate(test_loader, classes)

# %%
