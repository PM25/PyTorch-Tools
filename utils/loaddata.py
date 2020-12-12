from torch import from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset


class LoadData:
    def __init__(self, dataset=None, X_y=None):
        # default convert data to float
        if X_y != None:
            X, y = X_y
            X = from_numpy(X).float()
            y = from_numpy(y).float()
            dataset = TensorDataset(X, y)

        self.dataset = dataset

    def get_dataloader(self, split_ratio=[1], batch_size=128):
        dataloaders = []
        for splitted_dataset in self.split(split_ratio):
            dataloaders.append(
                DataLoader(splitted_dataset, batch_size=batch_size, shuffle=True)
            )
        if len(dataloaders) == 1:
            dataloaders = dataloaders[0]
        return dataloaders

    def split(self, ratio=[0.7, 0.2, 0.1]):
        counts = []
        for r in ratio:
            counts.append(int(len(self.dataset) * r))

        if sum(counts) != len(self.dataset):
            missing_count = len(self.dataset) - sum(counts)
            counts[-1] += missing_count

        return random_split(self.dataset, counts)
