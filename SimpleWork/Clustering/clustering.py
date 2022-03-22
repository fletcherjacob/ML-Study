
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
from sklearn.datasets import make_blobs



X,y  = make_blobs(n_samples = 100, centers = 5)
plt.scatter(X[:,0],X[:,1])
plt.show()



X,y  = make_blobs(n_samples = 500, centers = 5)
plt.scatter(X[:,0],X[:,1])
plt.show()

#note how visually the clusters decrease as n_samples increase in values

 X,y  = make_blobs(n_samples = 500, centers = 5, cluster_std = 2)
 plt.scatter(X[:,0],X[:,1])
 plt.show()

 #note how the clusters converge on one another.

X,y  = make_blobs(n_samples = 500, centers = 5, cluster_std = 0.6)
plt.scatter(X[:,0],X[:,1])
plt.show()

 # By reducing the the standard deeviation of the clusters, they now become more defined

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5) #create KMeans model
kmeans.fit(X) #train the model
y_pred = kmeans.predict(X)
plt.scatter(X[:,0],X[:,1], c=y_pred, cmap='viridis')
