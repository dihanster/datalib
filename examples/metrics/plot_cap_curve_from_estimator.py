"""
    Module with plot example for CAP curve display.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datalib.metrics import CAPCurveDisplay

X, y = make_classification(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = SVC(random_state=0).fit(X_train, y_train)

CAPCurveDisplay.from_estimator(clf, X_test, y_test)

plt.show()
