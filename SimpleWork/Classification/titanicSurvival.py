import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC  # import Support Vector Classifier module

#import Data
titanicData = pd.read_csv(
    r"/home/fletcher/Repos/ML-Study/SimpleWork/Classification/train.csv")
print(titanicData)

#Repalced Survived Binary to String
print(titanicData['Survived'])
titanicData['Survived'] = titanicData['Survived'].replace(1, 'Survived')
titanicData['Survived'] = titanicData['Survived'].replace(0, 'Died')
print(titanicData['Survived'])

# exploratory analysis of the dataset

#histogram of age
titanicData['Age'].plot.hist(edgecolor='black')
plt.show()


#Pie Charts for percentage breakdown of each category
plt.figure(0)
titanicData['Sex'].value_counts().plot.pie()

plt.figure(1)
titanicData['Pclass'].value_counts().plot.pie()

plt.figure(3)
titanicData['Survived'].value_counts().plot.pie()

plt.show()

#Feature Extraction , Seperate Features from Labels
#Missing Value Check and Drop function
print(titanicData.isnull().sum())
titanicData = titanicData.dropna()


#Hot Encoding for Sex Feature
X = pd.get_dummies(titanicData, columns=['Sex'])

#Split Dataset into Label and Features
X = X.drop(columns='Survived')
y = titanicData['Survived']


#Get Feature names
feature_names = X.keys()

#Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#create Classifier model
rfc = RandomForestClassifier()

#train the classifier
rfc.fit(X_train, y_train)

#test the classifier
y_pred_rfc = rfc.predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred_rfc))


#Display feature importance  of Random Tree Classifier
feature_imp_ = pd.Series(rfc.feature_importances_,
                         index=feature_names).sort_values(ascending=False)

sort = rfc.feature_importances_.argsort()
plt.barh(feature_names[sort], rfc.feature_importances_[sort])
plt.xlabel("Feature Importance")
plt.show()


svc = SVC(gamma='auto')  # Create a Support Vector Classifier
svc.fit(X_train, y_train)  # Fit the classifier to the input training data
y_pred_svc = svc.predict(X_test)


print("ACCURACY OF THE RANDOM Forest MODEL: ",
      metrics.accuracy_score(y_test, y_pred_rfc))
print("ACCURACY OF THE Support Vector MODEL: ",
      metrics.accuracy_score(y_test, y_pred_svc))
