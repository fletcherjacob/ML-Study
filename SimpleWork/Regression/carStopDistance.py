import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#import Data
dataset= pd.read_csv(r'/home/fletcher/Repos/ML-Study/SimpleWork/Regression/stopDistance.csv')

#quick null Check
print(dataset.isnull().sum())

#split data
y = dataset.drop(columns='Speed')
X = dataset.drop(columns='Distance')

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#MODEL
regr = linear_model.LinearRegression()


#Train the MODEL
regr.fit(X_train,y_train)

#Make predictions using the testing set
y_pred = regr.predict(X_test)


# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Plot outputs
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
