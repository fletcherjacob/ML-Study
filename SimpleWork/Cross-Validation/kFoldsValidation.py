from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits


#import sklearn digit image data set
digits = load_digits()



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size = 0.3)


#compare multiple models accuracies
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)


svm =SVC()
svm.fit(X_train,y_train)
svm.score(X_test, y_test)


rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf.score(X_test, y_test)


#kfold
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)  #specify amount of folds

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

###############################################################################
def get_score(model,X_train,X_test,y_train,y_test ):  #simplified score function
    model.fit(X_train,y_train)
    return model.score(X_test, y_test)



###Complex kfold
from sklearnmodel_selection import StratifiedKFold #used in classification and
#keeps the target classes ration the same for each StratifiedKFold

fold = StratifiedKFold(n_splits = 3)


#create lists to store model scores
scores_lr = []
scores_svm = []
scores_rf = []


for train_index, test_index in kf.split(digits.data):
    X_train,X_test,y_train,y_test = digits.data[train_index], digits.data[test_index],\
                                    digits.target[train_index],digits.target[test_index]
    scores_lr.append(get_score(LogisticRegression(),X_train,X_test, y_train, y_test))    #add score results to score lists
    scores_svm.append(get_score(SVC(),X_train,X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(),X_train,X_test, y_train, y_test) )

print("Linear Regression scores: " , scores_lr)
print("SVM scores: " , scores_svm)
print("Random Forrest scores: " , scores_rf)

#simple crossfold
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),digits.data, digits.target)
