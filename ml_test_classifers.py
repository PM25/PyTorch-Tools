#%%
from utils import *
from datapreprocessing import processing_data

from sklearn.model_selection import train_test_split

# start from here!
if __name__ == "__main__":
    (X_np, y_np) = processing_data(binary=True)

    mlmodelwrapper = MLModelWrapper(X_np, y_np)
    mlmodelwrapper.test_classifiers()
