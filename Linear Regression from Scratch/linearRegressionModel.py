import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


n = 20
X = np.arange(n)
y = 4 * X + 3
plt.scatter(X, y)

on = np.ones((X.size, 1))

print(X.shape)
print(on.shape)

# note how the shapes are different a new axis must be added when trying to conecate the arrays
X2 = np.hstack((X[:, np.newaxis], on))

y = y[:, np.newaxis]


a = np.linalg.lstsq(X2, y, rcond=None) #least square function

print(a[0]) # note how the provided values

y_pred = a[0][0] * X + a[0][1]
