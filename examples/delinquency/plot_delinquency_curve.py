from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datalib import DeliquencyDisplay


"""
==================
Delinquency curves
==================
The delinquency analysis is key to understand the default rate pondered
by approval rates. This example demonstrates how leverage deliquency curves for
 a binary classifier.
"""

X, y = make_classification(
    n_samples=100, n_features=5, n_informative=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)

clf = LogisticRegression()

clf.fit(X_train, y_train)
display = DeliquencyDisplay.from_estimator(clf, X_test, y_test)
display.show()
