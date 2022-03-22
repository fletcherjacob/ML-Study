
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
from sklearn.datasets import make_blobs
sbs.set()


A = sbs.load_dataset('iris')
print(A)

print(A.head())#shows default of the 1st 5 rows  Note that the first 4 columns are the features and the last is the class category label for the sample
print(A.tails())#shows default of the last 5 rows
A.iloc[60:65,:] #displays sample rows 60-64 within the data dataset
