#%%
from utils import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


train_X = pd.read_csv("data/train.csv")
# y = pd.read_csv("data/train_label.csv")

pd_y = train_X["reservation_status"].astype("category")
cats = pd_y.cat.categories
print(cats)
pd_y = pd_y.cat.codes
pd_X = train_X.iloc[
    :,
    ~train_X.columns.isin(
        [
            "is_canceled",
            "ID",
            "adr",
            "reservation_status",
            "reservation_status_date",
            "hotel",
        ]
    ),
]

get_columns_with_nan(pd_X)

#%%
pd_X["children"] = pd_X["children"].fillna(0)
nan_cols = list(get_columns_with_nan(pd_X))
pd_X[nan_cols] = None
for col in pd_X.select_dtypes(include=["object"]).columns:
    pd_X[col] = pd_X[col].factorize()[0]

print(f"NaN colums: {get_columns_with_nan(pd_X)}")

numpy_X = pd_X.to_numpy()
numpy_y = pd_y.to_numpy()

X, y = numpy_X, numpy_y
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25, random_state=0
)


#%%
if __name__ == "__main__":
    mlmodelwrapper = MLModelWrapper(train_X, train_y, test_X, test_y)
    mlmodelwrapper.test_classifiers()
