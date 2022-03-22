import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

n = 20
X = np.arange(n)#creates evenly spaced sample values
y =  4*X  + 3*(X**2) - 100 #funtction for Y value

print(X)
print(y)
plt.scatter(X, y,)#plots coordinates
plt.title('Regression') # provides Title
plt.show() # displays the plot

#####################################################################333
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.linear_model import LinearRegression

n = 20
X = np.arange(n)#creates evenly spaced sample values
y =  4*X  + 3*(X**2) - 100 #funtction for Y value

m = LinearRegression()
m.fit(X[;,np.newaxis], y)
y_pred = m.predict(X[;,np.newaxis])
plt.scatter(X,y)
plt.plot(X , y_pred , color = 'red')
plt.show()

########################################################

#Flexible preditcions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

n = 20
X = np.arange(n)#creates evenly spaced sample values
y =  4*X  + 3*(X**2) - 100 #funtction for Y value

polyModel = PolynomialFeatures(degree=2)
X_poly = polyModel.fit_transform(X[:,np.newaxis])
m = LinearRegression() # Create linear regression object
m.fit(X_poly, y) #train the model
y_pred = m.predict(X_poly) #predictions


plt.scatter(X,y)
plt.plot(X,y_pred, color = 'red')
plt.show()
