"""
Module with the test for the bootstraps wraping metrics functions.
"""

import pytest

from sklearn import datasets
from sklearn import metrics as sk_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_random_state

from .._bootstrap import bootstrap_metric


# iris
iris = datasets.load_iris()


@pytest.mark.parametrize(
    "metric, threshold_dependent, kwargs",
    [
        (sk_metrics.roc_auc_score, False, {}),
        (sk_metrics.average_precision_score, False, {}),
        (sk_metrics.accuracy_score, True, {}),
        (sk_metrics.f1_score, True, {}),
        (sk_metrics.fbeta_score, True, {"beta": 3}),
    ],
)
@pytest.mark.parametrize(
    "sample_weight",
    [None, check_random_state(42).exponential(size=iris.target.shape[0])],
)
def test_bootstrap_metric_binary_classification(
    metric, threshold_dependent, kwargs, sample_weight
):
    # TODO: regression, multiclass...
    X, y = iris.data, (iris.target >= 1).astype(int)
    model = LogisticRegression(random_state=42).fit(X, y)

    if threshold_dependent:
        pred = model.predict(X)
    else:
        pred = model.predict_proba(X)[:, 1]

    # TODO: Ok. it is running, but what do I want to assert here?
    bootstrap_metric(y, pred, metric, sample_weight=sample_weight, **kwargs)

