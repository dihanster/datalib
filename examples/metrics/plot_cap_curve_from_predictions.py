"""
    Module with plot example for CAP curve display.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datalib.metrics import CAPCurveDisplay

X, y = make_classification(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(random_state=0).fit(X_train, y_train)

CAPCurveDisplay.from_predictions(y_test, clf.predict_proba(X_test)[:, 1])

plt.show()
