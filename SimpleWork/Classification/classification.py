import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
from sklearn.datasets import make_blobs
sbs.set()


X = sbs.load_dataset('iris')
print(X)
print(X.head())#shows default of the 1st 5 rows  Note that the first 4 columns are the features and the last is the class category label for the sample

y = X.species # pull labels from dataset
print(y.head())
print(y.shape)

X = X.drop(columns={'species'}) #remove  labels from data set
print(X.head())
print(X.shape)

from sklearn.svm import SVC #import Support Vector Classifier module

model = SVC(gamma='auto') # Create a Support Vector Classifier

model.fit(X,y) # Fit the classifier to the input training data

y_pred = model.predict(X)

print(y_pred)
