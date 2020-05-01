from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from sklearn import preprocessing

iris = datasets.load_iris()

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(iris.data)

#RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)     

# Test our classifier
scores = cross_val_score(clf, X, iris.target, cv=5)

print("Accuracy of Random Forest Classifier: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))

#DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()     

# Test our classifier
scores = cross_val_score(clf, X, iris.target, cv=5)

print("Accuracy of Decision Tree classifier: %0.2f +/- %0.2f" % (scores.mean(), scores.std() * 2))
