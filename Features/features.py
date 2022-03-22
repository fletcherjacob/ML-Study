import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set()


from sklearn.datasets import make_blobs


print(" Imports Loaded")
X,y = make_blobs(n_samples = 100, centers = 3) #creates gaussian blobs; n_samples is amount of points; centers is the amount fixed center locations;
print(X.shape)
print(type(X))
print(X[0,:]) #shows value of the first feature
print(y)


#IRIS dataset. 4 Attributes(features), 150 samples

A = sbn.load_dataset('iris')
print(A)

print(A.head())#shows default of the 1st 5 rows  Note that the first 4 columns are the features and the last is the class category label for the sample
print(A.tails())#shows default of the last 5 rows

A.iloc[60:65,:]
