# %%
from utils import *
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#%%
def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    # print(f"Columns with NaN: {columns_with_nan}")
    return columns_with_nan


#%%
train_X = pd.read_csv("data/train.csv")
# y = pd.read_csv("data/train_label.csv")

pd_y = train_X["is_canceled"]
pd_X = train_X.iloc[
    :,
    ~train_X.columns.isin(
        [
            "is_canceled",
            "ID",
            "adr",
            "reservation_status",
            "reservation_status_date",
            "country",
        ]
    ),
]
get_columns_with_nan(pd_X)

#%%
pd_X["children"] = pd_X["children"].fillna(0)
nan_cols = list(get_columns_with_nan(pd_X))
pd_X.iloc[:, pd_X.columns.isin(nan_cols)] = pd_X.iloc[
    :, pd_X.columns.isin(nan_cols)
].astype(str)
pd_X = pd.get_dummies(pd_X)
print(get_columns_with_nan(pd_X))

numpy_X = pd_X.to_numpy()
numpy_y = pd_y.to_numpy()


#%%
train_loader, val_loader, test_loader = LoadData(X_y=(numpy_X, numpy_y)).get_dataloader(
    [0.7, 0.2, 0.1], batch_size=64
)


# %% start from here!
if __name__ == "__main__":
    # setting
    model = BinaryClassificationModel(numpy_X.shape[1])
    loss_func = nn.BCELoss()
    # loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    modelwrapper = ModelWrapper(model, loss_func, optimizer)

    # training
    model = modelwrapper.train(train_loader, val_loader, max_epochs=1)
    # # resume training
    # modelwrapper.train(train_loader, val_loader, max_epochs=5)

    model.cpu()
    pred_y = model(torch.from_numpy(numpy_X).float())
    dataset = pd.DataFrame()
    dataset["is_cancel"] = numpy_y

    dataset2 = pd.DataFrame(pred_y.detach().numpy())

    tmp_df = train_X.iloc[
        :,
        ~train_X.columns.isin(
            [
                "is_canceled",
                "ID",
                "adr",
                "reservation_status",
                "reservation_status_date",
                "country",
            ]
        ),
    ]
    results_df = pd.concat([tmp_df, dataset, dataset2], axis=1)
    results_df.to_csv("result.csv", index=False)
    # # evaluate the model
    print(f"\ntest loss: {modelwrapper.validation(test_loader)}")
    modelwrapper.binary_classification_evaluate(test_loader, ["Not Cancel", "Cancel"])

# %%
