#@title Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Imported modules.")

train_df = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(
    train_df.index))  # shuffle the examples
test_df = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
